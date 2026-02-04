from typing import Sequence, Dict, Any, Optional, List
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import Counter

from mmpose.evaluation.metrics.keypoint_2d_metrics import PCKAccuracy
from mmdet.registry import METRICS
from tools.graph_grouping import (
    group_keypoints_into_instances,
    make_check_merge_max_label,
)
from tools.graph_matching import sequential_matching
from mmengine.logging import MMLogger
from mmpose.evaluation.functional import keypoint_pck_accuracy


@METRICS.register_module()
class GroupedPCKAccuracy(PCKAccuracy):
    """
    Group keypoints into instances using a per-label max derived from GT categories,
    then match to GT and apply graph matching (as in TopDownPCKAccuracy), and finally
    append padded arrays for PCK computation.
    """

    def __init__(self,
                 thr: float = 0.05,
                 node_score_thresh: float = 0.5,
                 edge_score_thresh: float = 0.5,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs):
        super().__init__(thr=thr,
                         norm_item='bbox',
                         collect_device=collect_device,
                         prefix=prefix)
        self.node_score_thresh = node_score_thresh
        self.edge_score_thresh = edge_score_thresh

    def _per_label_max_from_gt(self,
                               meta: Dict[str, Any],
                               anns: List[dict],
                               pred_label_set: set) -> Dict[str, int]:
        """
        Count keypoint label multiplicities from GT categories using meta['category_keypoints'].
        If categories share labels, take the maximum across categories. Only keep labels present
        in current predictions (pred_label_set)
        """
        cat_to_kpts = meta.get('category_keypoints', {})
        
        # Use categories appearing in the image
        cat_ids = {ann.get('category_id', -1) for ann in anns}
        max_per_label: Dict[str, int] = {}
        for cid in cat_ids:
            names = cat_to_kpts.get(cid, [])
            cnt = Counter(names)
            for name, c in cnt.items():
                if name in pred_label_set:
                    max_per_label[name] = max(max_per_label.get(name, 0), int(c))
        
        return max_per_label

    def process(self, data_batch: Sequence[dict], data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            if 'pred_instances' not in data_sample or 'raw_ann_info' not in data_sample:
                continue
            pred_inst = data_sample['pred_instances']
            anns: List[dict] = data_sample['raw_ann_info']
            if not isinstance(anns, list) or len(anns) == 0:
                continue

            meta = self.dataset_meta

            # Predicted nodes
            label_names_full: List[str] = list(pred_inst['label_names'])
            scores_full = pred_inst['scores'].cpu().numpy().astype(np.float32)
            coords_full = pred_inst['keypoints'].cpu().numpy().astype(np.float32)  # [N,2] global coords
            rel_full = pred_inst['relation_scores'].cpu().numpy().astype(np.float32)  # [N,N,C]
            N = scores_full.shape[0]
            if N == 0:
                continue

            # Compute per-label maxima from GT categories via meta
            pred_label_names_set = sorted(set(label_names_full))
            name_to_id = {n: i for i, n in enumerate(pred_label_names_set)}
            max_per_label_name = self._per_label_max_from_gt(
                meta=meta,
                anns=anns,
                pred_label_set=set(pred_label_names_set),
            )
            # only include labels that exist in GT-derived maxima
            per_label_max_ids = {
                name_to_id[n]: max_per_label_name[n]
                for n in pred_label_names_set
                if n in max_per_label_name
            }

            # Filter nodes by score
            node_mask = scores_full >= self.node_score_thresh
            if not np.any(node_mask):
                continue
            valid_idx = np.nonzero(node_mask)[0]
            f_labels = np.array([name_to_id[n] for n in np.array(label_names_full, dtype=object)[valid_idx]], dtype=int)
            f_scores = scores_full[valid_idx]
            f_coords = coords_full[valid_idx]
            f_rel = rel_full[np.ix_(valid_idx, valid_idx, np.arange(rel_full.shape[2]))]

            # Group nodes into instances with per-label maxima
            check_fn = make_check_merge_max_label(per_label_max_ids, relation_scores=f_rel)
            instance_groups = group_keypoints_into_instances(
                keypoint_labels=f_labels,
                keypoint_scores=f_scores,
                relation_scores=f_rel,
                check_merge=check_fn,
                min_edge_score=self.edge_score_thresh,
            )

            # Prepare per-group data
            group_coords: List[np.ndarray] = []
            group_scores: List[np.ndarray] = []
            group_rels: List[np.ndarray] = []
            group_centroids: List[np.ndarray] = []
            group_label_names: List[List[str]] = []

            for grp in instance_groups:
                # Map grp.node_ids (original indices) -> local indices in filtered arrays
                local_positions = []
                for nid in getattr(grp, 'node_ids', []):
                    pos = np.where(valid_idx == nid)[0]
                    if pos.size > 0:
                        local_positions.append(int(pos[0]))
                if len(local_positions) == 0:
                    continue

                idx = np.array(local_positions, dtype=int)
                coords = f_coords[idx]        # [M,2]
                scores = f_scores[idx]        # [M]
                rel = f_rel[np.ix_(idx, idx, np.arange(f_rel.shape[2]))]  # [M,M,C]
                labels_ids = f_labels[idx]
                labels_names = [pred_label_names_set[i] for i in labels_ids]

                # Symmetrize and zero diagonal per channel
                rel = (rel + np.transpose(rel, (1, 0, 2))) / 2.0
                for ch in range(rel.shape[2]):
                    np.fill_diagonal(rel[..., ch], 0.0)

                group_coords.append(coords)
                group_scores.append(scores)
                group_rels.append(rel)
                group_centroids.append(coords.mean(axis=0))
                group_label_names.append(labels_names)

            if len(group_coords) == 0:
                continue

            pred_centroids = np.stack(group_centroids, axis=0)  # [P,2]

            # Build GT centers and per-ann info
            gt_centers, gt_sides, gt_ids, gt_cat_ids = [], [], [], []
            for ann in anns:
                x, y, w, h = ann['bbox']
                side = max(w, h)
                center = np.array([x + 0.5 * w, y + 0.5 * h], dtype=np.float32)
                gt_centers.append(center)
                gt_sides.append(side)
                gt_ids.append(ann.get('id', -1))
                gt_cat_ids.append(ann.get('category_id', -1))
            if len(gt_centers) == 0:
                continue
            gt_centers = np.stack(gt_centers, axis=0)  # [G,2]

            # Hungarian assignment on centroid distance
            cost = np.linalg.norm(pred_centroids[:, None, :] - gt_centers[None, :, :], axis=-1)  # [P,G]
            row_ind, col_ind = linear_sum_assignment(cost)

            # For each matched pair, use skeleton-aware graph matching (same as topdown)
            for r, c in zip(row_ind, col_ind):
                ann = anns[c]
                cat_id = ann['category_id']
                # Build GT keypoints and relation matrix from meta skeleton
                gt_label_names_all = list(meta['category_keypoints'][cat_id])
                gt_kpts_all = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
                min_len = min(len(gt_label_names_all), gt_kpts_all.shape[0])
                gt_label_names_all = gt_label_names_all[:min_len]
                gt_kpts_all = gt_kpts_all[:min_len]

                gt_rel_full = np.zeros((gt_kpts_all.shape[0], gt_kpts_all.shape[0], 1), dtype=np.float32)
                skel = meta['category_skeleton'][cat_id]
                for u, v in skel:
                    if u != v and u < gt_rel_full.shape[0] and v < gt_rel_full.shape[0]:
                        gt_rel_full[u, v, 0] = 1.0
                        gt_rel_full[v, u, 0] = 1.0

                visible = gt_kpts_all[:, 2] > 0
                gt_kpts = gt_kpts_all[visible]
                gt_rel = gt_rel_full[visible][:, visible, :]
                gt_label_names = [n for i, n in enumerate(gt_label_names_all) if visible[i]]
                Kg = gt_kpts.shape[0]
                if Kg == 0:
                    continue

                # Pred group
                pred_coords_grp = group_coords[r]           # [Mp,2] already global
                pred_scores_grp = group_scores[r]           # [Mp]
                pred_rel = group_rels[r]                    # [Mp,Mp,C]
                pred_names = group_label_names[r]           # [Mp]

                # Labels mapping
                all_names = sorted(set(gt_label_names) | set(pred_names))
                name_to_idx_l = {n: i for i, n in enumerate(all_names)}
                gt_labels = np.array([name_to_idx_l[n] for n in gt_label_names], dtype=int)
                pred_labels = np.array([name_to_idx_l[n] for n in pred_names], dtype=int)

                match = sequential_matching(
                    algorithm_sequence=['rrwm', 'our2opt'],
                    gt_labels=gt_labels,
                    gt_relation_matrices=gt_rel,
                    pred_labels=pred_labels,
                    pred_scores=pred_scores_grp,
                    pred_relation_matrices=pred_rel,
                )

                # Variable-length arrays aligned to GT order (no padding)
                gt_coords = gt_kpts[:, :2].reshape(1, Kg, 2)
                pred_coords = np.zeros_like(gt_coords, dtype=np.float32)
                for gt_pos, pred_pos in match.items():
                    if 0 <= gt_pos < Kg and 0 <= pred_pos < pred_coords_grp.shape[0]:
                        pred_coords[0, gt_pos, 0] = pred_coords_grp[pred_pos, 0]
                        pred_coords[0, gt_pos, 1] = pred_coords_grp[pred_pos, 1]

                # Mask over visible GT keypoints
                mask = (gt_kpts[:, 2] > 0).reshape(1, Kg)

                side = float(max(ann['bbox'][2], ann['bbox'][3]))
                self.results.append({
                    'pred_coords': pred_coords,
                    'gt_coords': gt_coords,
                    'mask': mask,
                    'bbox_size': np.array([[side, side]], dtype=np.float32),
                    'gt_instance_id': gt_ids[c],
                    'category_id': gt_cat_ids[c],
                })

    def compute_metrics(self, results: list) -> Dict[str, float]:
        # Group by category and compute macro-averaged PCK over categories
        logger: MMLogger = MMLogger.get_current_instance()

        # Build category -> indices mapping
        cat_to_idxs: Dict[int, List[int]] = {}
        for i, r in enumerate(results):
            cid = r.get('category_id', -1)
            cat_to_idxs.setdefault(cid, []).append(i)

        metrics: Dict[str, float] = {}
        per_cat_pck: List[float] = []

        for cid, idxs in cat_to_idxs.items():
            # Flatten all keypoints across samples in this category
            pred_rows, gt_rows, mask_rows, norm_rows = [], [], [], []
            for i in idxs:
                pc = results[i]['pred_coords'][0]    # (K, 2)
                gc = results[i]['gt_coords'][0]      # (K, 2)
                m = results[i]['mask'][0].astype(bool)  # (K,)
                if pc.shape[0] == 0:
                    continue
                pred_rows.append(pc)
                gt_rows.append(gc)
                mask_rows.append(m)
                norm = results[i]['bbox_size'][0]    # (2,)
                norm_rows.append(np.repeat(norm.reshape(1, 2), pc.shape[0], axis=0))

            if not pred_rows:
                continue

            pred_all = np.concatenate(pred_rows, axis=0)   # (M, 2)
            gt_all = np.concatenate(gt_rows, axis=0)       # (M, 2)
            mask_all = np.concatenate(mask_rows, axis=0)   # (M,)
            norm_all = np.concatenate(norm_rows, axis=0)   # (M, 2)

            # Each keypoint as an instance (N=M, K=1)
            pred_flat = pred_all.reshape(-1, 1, 2)
            gt_flat = gt_all.reshape(-1, 1, 2)
            mask_flat = mask_all.reshape(-1, 1)

            logger.info(f'Evaluating {self.__class__.__name__} for category {cid} (normalized by "bbox_size")...')
            _, pck, _ = keypoint_pck_accuracy(pred_flat, gt_flat, mask_flat, self.thr, norm_all)
            metrics[f'PCK/cat_{cid}'] = pck
            per_cat_pck.append(pck)

        # Macro-average over categories
        metrics['PCK'] = float(np.mean(per_cat_pck)) if per_cat_pck else 0.0
        logger.info(f'Macro-averaged PCK over {len(per_cat_pck)} categories: {metrics["PCK"]:.4f}')

        return metrics
