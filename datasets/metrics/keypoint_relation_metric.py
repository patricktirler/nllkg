from typing import Dict, Optional, Sequence
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from collections import OrderedDict
import os
import json
from mmdet.registry import METRICS
from .utils import (
    compute_pairwise_distances,
    compute_precision,
    compute_recall,
    compute_precision_recall_relation,
)


@METRICS.register_module()
class KeypointRelationMetric(BaseMetric):
    """Evaluation metric for open vocabulary pose estimation with relation prediction.
    
    Args:
        keypoint_score_threshold (float): Fixed keypoint score threshold for recall@distance and relation metrics. Defaults to 0.3.
        keypoint_distance_threshold (float): Fixed distance threshold for keypoint PR/F1 vs score. Defaults to 0.005.
        keypoint_score_thresholds (Sequence[float]): Score thresholds to sweep for keypoint PR/F1. Defaults to (0.3, 0.5, 0.7).
        relation_score_threshold (float): Relation score threshold for precision/recall/F1 computation. Defaults to 0.3.
        distance_thresholds (Sequence[float]): Distance thresholds for recall@distance.
            Values in [0, 1] relative to max(img_w, img_h). Defaults to (0.001, 0.005, 0.01, 0.05).
        collect_device (str): Device for collecting results ('cpu' or 'gpu'). Defaults to 'cpu'.
        prefix (str, optional): Prefix for metric names. Defaults to None.
        save_path_preds (str, optional): Path to save prediction results. Defaults to None.
        crop_sizes (Sequence[str], optional): List of target crop sizes as tuples (height, width)
            (e.g., [(2400, 3999), (800, 1333)]).
            If provided, all occurring crop sizes are mapped to the closest target size by area.
            If None (default), uses all crop sizes found in the data.
    """
    default_prefix: Optional[str] = 'keypointgraph'

    def __init__(self,
                 keypoint_score_threshold: float = 0.3,
                 keypoint_distance_threshold: float = 0.005,
                 keypoint_score_thresholds: Sequence[float] = (0.3, 0.5, 0.7),
                 relation_score_threshold: float = 0.3,
                 distance_thresholds: Sequence[float] = (0.001, 0.005, 0.01, 0.05),
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 save_path_preds: str = None,
                 crop_sizes: Optional[Sequence[str]] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.distance_thresholds = tuple(distance_thresholds)
        self.keypoint_distance_threshold = float(keypoint_distance_threshold)
        self.keypoint_score_thresholds = tuple(keypoint_score_thresholds)
        self.keypoint_score_threshold = float(keypoint_score_threshold)
        self.relation_score_threshold = float(relation_score_threshold)
        self.save_path_preds = save_path_preds
        if crop_sizes is not None:
            normalized_sizes = []
            for c in crop_sizes:
                if isinstance(c, (tuple, list)):
                    # Convert tuple/list (height, width) to string "WxH" format (to match get_crop_size output)
                    normalized_sizes.append(f"{int(c[1])}x{int(c[0])}")
                else:
                    # Already a string, just normalize
                    normalized_sizes.append(str(c))
            self.crop_sizes = tuple(normalized_sizes)
        else:
            self.crop_sizes = None

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred = data_sample['pred_instances']
            img_w = int(data_sample['ori_shape'][1])
            img_h = int(data_sample['ori_shape'][0])
            max_dim = max(img_w, img_h)

            # Handle cropped images
            crop_bbox = data_sample.get('crop_bbox', None)
            gt_coords = data_sample['gt_instances']['keypoint_coords'].cpu().numpy().reshape(-1, 2)
            gt_label_names = np.array([data_sample['text'][label] for label in data_sample['gt_instances']['labels']])
            gt_relations = data_sample['gt_instances']['relation_matrices'].cpu().numpy()
            pred_coords = pred['keypoints'].cpu().numpy().reshape(-1, 2)
            
            if crop_bbox is not None:
                # crop_bbox format: [x1, y1, x2, y2]
                x1, y1, x2, y2 = crop_bbox
                
                # Filter GT keypoints: keep only those inside crop region
                gt_mask = (
                    (gt_coords[:, 0] >= x1) & (gt_coords[:, 0] < x2) &
                    (gt_coords[:, 1] >= y1) & (gt_coords[:, 1] < y2)
                )
                gt_coords = gt_coords[gt_mask]
                gt_label_names = gt_label_names[gt_mask]
                
                # Filter relation matrices
                gt_relations = gt_relations[gt_mask][:, gt_mask, :]
                
                # Translate predicted keypoints to global coordinates
                pred_coords[:, 0] += x1
                pred_coords[:, 1] += y1
            
            record = dict(
                img_id=data_sample['img_id'],
                img_path=data_sample['img_path'],
                img_width=img_w,
                img_height=img_h,
                max_dim=max_dim,
                gt_coords=gt_coords,
                gt_label_names=gt_label_names,
                gt_relations=gt_relations,
                relation_names=data_sample.get('relation_text', []),
                pred_coords=pred_coords,
                pred_label_names=np.array(pred['label_names']),
                pred_scores=pred['scores'].cpu().numpy(),
                pred_relations=pred['relation_scores'].cpu().numpy(),
                crop_bbox=crop_bbox,
            )
            self.results.append(record)
            
            if self.save_path_preds is not None:
                os.makedirs(self.save_path_preds, exist_ok=True)
                img_filename = os.path.splitext(os.path.basename(data_sample['img_path']))[0]
                pred_dict = {
                    'img_id': record['img_id'],
                    'img_path': record['img_path'],
                    'keypoint_label_names': pred['label_names'],
                    'keypoint_labels': pred.get('labels', None) and pred['labels'].cpu().tolist(),
                    'keypoint_scores': record['pred_scores'].tolist(),
                    'keypoint_coords': pred_coords.tolist(),
                    'keypoint_relation_scores': record['pred_relations'].tolist() if record['pred_relations'] is not None else None,
                }
                gt_dict = {
                    'img_id': record['img_id'],
                    'img_path': record['img_path'],
                    'img_width': record['img_width'],
                    'img_height': record['img_height'],
                    'keypoint_label_names': record['gt_label_names'].tolist(),
                    'keypoint_coords': record['gt_coords'].tolist(),
                    'relation_matrices': record['gt_relations'].tolist() if record['gt_relations'] is not None else None,
                    'relation_names': record['relation_names'],
                }
                with open(os.path.join(self.save_path_preds, f"{img_filename}_pred.json"), "w") as f:
                    json.dump(pred_dict, f, indent=2)
                with open(os.path.join(self.save_path_preds, f"{img_filename}_gt.json"), "w") as f:
                    json.dump(gt_dict, f, indent=2)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        logger.info("Computing keypoint and relation metrics ...")
        logger.info("")

        # Pack lists
        gt_coords_list = [r['gt_coords'] for r in results]
        gt_label_names_list = [r['gt_label_names'] for r in results]
        pred_coords_list = [r['pred_coords'] for r in results]
        pred_label_names_list = [r['pred_label_names'] for r in results]
        pred_scores_list = [r['pred_scores'] for r in results]
        max_dim_list = [r['max_dim'] for r in results]

        gt_relations_list = [r['gt_relations'] for r in results]
        relation_names_list = [r['relation_names'] for r in results]
        pred_relations_list = [r['pred_relations'] for r in results]

        distances_list = [
            compute_pairwise_distances(gt, pred)
            for gt, pred in zip(gt_coords_list, pred_coords_list)
        ]

        metrics = OrderedDict()

        # 1) Keypoint PR/F1 vs score thresholds at fixed distance
        score_thr_vec = np.array(self.keypoint_score_thresholds, dtype=float)
        kp_prec = compute_precision(
            gt_coords_list=gt_coords_list,
            gt_labels_list=gt_label_names_list,
            pred_coords_list=pred_coords_list,
            pred_labels_list=pred_label_names_list,
            pred_scores_list=pred_scores_list,
            max_dim_list=max_dim_list,
            distance_threshold_norm=self.keypoint_distance_threshold,
            score_threshold=score_thr_vec,
            distances_list=distances_list,
        )
        kp_rec = compute_recall(
            gt_coords_list=gt_coords_list,
            gt_labels_list=gt_label_names_list,
            pred_coords_list=pred_coords_list,
            pred_labels_list=pred_label_names_list,
            pred_scores_list=pred_scores_list,
            max_dim_list=max_dim_list,
            distance_threshold_norm=self.keypoint_distance_threshold,
            score_threshold=score_thr_vec,
            distances_list=distances_list,
        )
        kp_prec = np.array(kp_prec)
        kp_rec = np.array(kp_rec)
        kp_thr = score_thr_vec
        kp_f1 = np.where(
            (kp_prec + kp_rec) > 0,
            2 * kp_prec * kp_rec / (kp_prec + kp_rec),
            0.0
        )

        logger.info("Keypoints: Precision/Recall/F1 at fixed distance threshold %.4f (relative to max(img dim))", self.keypoint_distance_threshold)
        for s, p, r_, f in zip(kp_thr, kp_prec, kp_rec, kp_f1):
            logger.info("  score>=%.2f -> Precision:%.3f Recall:%.3f F1:%.3f", s, p, r_, f)
            metrics[f"keypoints/f1@score{float(s):.2f}"] = float(f)
        logger.info("")

        # 2) Keypoint recall vs distance thresholds at fixed keypoint score
        logger.info(
            "Keypoints: Recall@distance (score>=%.2f)",
            self.keypoint_score_threshold
        )

        # ------------------------------------------------------------
        # Build crop size list
        # ------------------------------------------------------------
        def get_crop_size(result):
            crop_bbox = result.get('crop_bbox', None)
            if crop_bbox is not None:
                x1, y1, x2, y2 = crop_bbox
                crop_w = int(x2 - x1)
                crop_h = int(y2 - y1)
                return f"{crop_w}x{crop_h}"
            else:
                return f"{result['img_width']}x{result['img_height']}"

        crop_size_array = np.array([get_crop_size(result) for result in results], dtype=object)
        unique_crop_sizes = np.unique(crop_size_array)

        # Map crop sizes to specified crop sizes if provided
        def get_crop_size_area(crop_size_str):
            w, h = map(int, crop_size_str.split("x"))
            return w * h

        if self.crop_sizes is not None:
            # Create a mapping from actual crop sizes to specified crop sizes based on area
            crop_size_mapping = {}
            for actual_size in unique_crop_sizes:
                actual_area = get_crop_size_area(actual_size)
                # Find the closest specified crop size by area
                closest_size = min(self.crop_sizes, key=lambda s: abs(get_crop_size_area(s) - actual_area))
                crop_size_mapping[actual_size] = closest_size
            
            # Remap the crop_size_array
            crop_size_array = np.array([crop_size_mapping[cs] for cs in crop_size_array], dtype=object)
            unique_crop_sizes = np.unique(crop_size_array)
            sorted_crop_sizes = sorted(self.crop_sizes, key=get_crop_size_area, reverse=True)
        else:
            crop_size_mapping = None
            def crop_area(crop_size_str):
                return get_crop_size_area(crop_size_str)
            sorted_crop_sizes = sorted(unique_crop_sizes, key=crop_area, reverse=True)

        # ------------------------------------------------------------
        # Collect all unique label names globally
        # ------------------------------------------------------------
        all_label_names_per_image = [set(np.unique(label_names)) for label_names in gt_label_names_list]
        unique_label_names_all = sorted(set().union(*all_label_names_per_image))

        # Convert lists to numpy arrays once
        gt_coords_array = np.array(gt_coords_list, dtype=object)
        gt_label_names_array = np.array(gt_label_names_list, dtype=object)
        pred_coords_array = np.array(pred_coords_list, dtype=object)
        pred_label_names_array = np.array(pred_label_names_list, dtype=object)
        pred_scores_array = np.array(pred_scores_list, dtype=object)
        max_dim_array = np.array(max_dim_list)
        distances_array = np.array(distances_list, dtype=object)

        # ------------------------------------------------------------
        # Sort thresholds for display
        # ------------------------------------------------------------
        sorted_dist_thresholds = sorted(self.distance_thresholds, reverse=True)
        sorted_dist_indices = [self.distance_thresholds.index(t) for t in sorted_dist_thresholds]

        # ------------------------------------------------------------
        # GLOBAL column widths (computed once)
        # ------------------------------------------------------------
        label_col_width = max(
            max(len(str(l)) for l in unique_label_names_all),
            len("Label"),
            10
        )

        dist_headers = [f"{t:.4f}" for t in sorted_dist_thresholds]
        col_widths = [max(len(h), 7) for h in dist_headers]

        for crop_size in sorted_crop_sizes:
            crop_mask = crop_size_array == crop_size
            crop_indices = np.where(crop_mask)[0]

            logger.info("  Crop size: %s", crop_size)

            crop_gt_coords_array = gt_coords_array[crop_indices]
            crop_gt_label_names_array = gt_label_names_array[crop_indices]
            crop_pred_coords_array = pred_coords_array[crop_indices]
            crop_pred_label_names_array = pred_label_names_array[crop_indices]
            crop_pred_scores_array = pred_scores_array[crop_indices]
            crop_max_dim_array = max_dim_array[crop_indices]
            crop_distances_array = distances_array[crop_indices]

            # ---- Header (same width for every crop) ----
            header = (
                "Label".ljust(label_col_width)
                + " | "
                + " | ".join([h.rjust(w) for h, w in zip(dist_headers, col_widths)])
            )
            logger.info("    %s", header)

            separator = (
                "-" * label_col_width
                + "-+-"
                + "-+-".join(["-" * w for w in col_widths])
            )
            logger.info("    %s", separator)

            # ---- Rows ----
            all_label_recalls = []  # Track recalls for averaging
            
            for label_name in unique_label_names_all:

                label_mask = np.array([label_name in np.unique(label_names) for label_names in crop_gt_label_names_array])
                if not np.any(label_mask):
                    continue

                label_local_indices = np.where(label_mask)[0]

                # Filter GT coordinates and labels to only include the target label_name
                label_gt_coords = []
                label_gt_label_names = []
                label_pred_coords = []
                label_pred_label_names = []
                label_pred_scores = []
                label_max_dim = []
                label_distances = []
                
                for idx in label_local_indices:
                    mask = crop_gt_label_names_array[idx] == label_name
                    label_gt_coords.append(crop_gt_coords_array[idx][mask])
                    label_gt_label_names.append(crop_gt_label_names_array[idx][mask])
                    label_pred_coords.append(crop_pred_coords_array[idx])
                    label_pred_label_names.append(crop_pred_label_names_array[idx])
                    label_pred_scores.append(crop_pred_scores_array[idx])
                    label_max_dim.append(crop_max_dim_array[idx])
                    label_distances.append(crop_distances_array[idx][mask])

                label_recalls = compute_recall(
                    gt_coords_list=label_gt_coords,
                    gt_labels_list=label_gt_label_names,
                    pred_coords_list=label_pred_coords,
                    pred_labels_list=label_pred_label_names,
                    pred_scores_list=label_pred_scores,
                    max_dim_list=label_max_dim,
                    distance_threshold_norm=self.distance_thresholds,
                    score_threshold=self.keypoint_score_threshold,
                    distances_list=label_distances,
                )

                label_recalls = np.asarray(label_recalls, dtype=float)
                all_label_recalls.append(label_recalls)

                recall_values = [f"{label_recalls[i]:.3f}" for i in sorted_dist_indices]
                recall_str = " | ".join(
                    [v.rjust(w) for v, w in zip(recall_values, col_widths)]
                )

                logger.info(
                    "    %s | %s",
                    str(label_name).ljust(label_col_width),
                    recall_str
                )

            # ---- Average row ----
            if all_label_recalls:
                separator = (
                    "-" * label_col_width
                    + "-+-"
                    + "-+-".join(["-" * w for w in col_widths])
                )
                logger.info("    %s", separator)
                
                avg_recalls = np.mean(all_label_recalls, axis=0)
                avg_recall_values = [f"{avg_recalls[i]:.3f}" for i in sorted_dist_indices]
                avg_recall_str = " | ".join(
                    [v.rjust(w) for v, w in zip(avg_recall_values, col_widths)]
                )
                logger.info(
                    "    %s | %s",
                    "AVERAGE".ljust(label_col_width),
                    avg_recall_str
                )

            logger.info("")

        logger.info("")

        # 3) Relations: Precision/Recall/F1 at fixed relation score threshold
        logger.info(
            "Relations: Precision/Recall/F1 (keypoint_score>=%.2f, dist<=%.4f, rel_score>=%.2f)",
            self.keypoint_score_threshold, self.keypoint_distance_threshold, self.relation_score_threshold
        )
        
        # Filter out images without relations or predictions
        if any(g is not None and p is not None for g, p in zip(gt_relations_list, pred_relations_list)):
            relation_curves = compute_precision_recall_relation(
                gt_coords_list=gt_coords_list,
                gt_labels_list=gt_label_names_list,
                gt_relations_list=gt_relations_list,
                relation_names_list=relation_names_list,
                pred_coords_list=pred_coords_list,
                pred_labels_list=pred_label_names_list,
                pred_scores_list=pred_scores_list,
                pred_relations_list=pred_relations_list,
                max_dim_list=max_dim_list,
                keypoint_score_threshold=self.keypoint_score_threshold,
                keypoint_distance_threshold=self.keypoint_distance_threshold,
                relation_score_thresholds=np.array([self.relation_score_threshold], dtype=float),
                distances_list=distances_list
            )
            
            if relation_curves:
                # Prepare table: Relation | Precision | Recall | F1
                rel_col_width = max(
                    max(len(str(rel)) for rel in relation_curves.keys()),
                    len("Relation"),
                    15
                )
                metric_col_width = 10
                
                # Header
                header = (
                    "Relation".ljust(rel_col_width)
                    + " | "
                    + "Precision".center(metric_col_width)
                    + " | "
                    + "Recall".center(metric_col_width)
                    + " | "
                    + "F1".center(metric_col_width)
                )
                logger.info("  %s", header)
                
                separator = (
                    "-" * rel_col_width
                    + "-+-"
                    + "-" * metric_col_width
                    + "-+-"
                    + "-" * metric_col_width
                    + "-+-"
                    + "-" * metric_col_width
                )
                logger.info("  %s", separator)
                
                # Rows for each relation
                all_prec = []
                all_rec = []
                all_f1 = []
                
                for rel in sorted(relation_curves.keys()):
                    curves = relation_curves[rel]
                    prec = float(curves['precision'][0])
                    rec = float(curves['recall'][0])
                    f1 = float(curves['f1'][0])
                    
                    all_prec.append(prec)
                    all_rec.append(rec)
                    all_f1.append(f1)
                    
                    logger.info(
                        "  %s | %s | %s | %s",
                        str(rel).ljust(rel_col_width),
                        f"{prec:.3f}".center(metric_col_width),
                        f"{rec:.3f}".center(metric_col_width),
                        f"{f1:.3f}".center(metric_col_width)
                    )
                    metrics[f"relations/{rel}/f1@score{self.relation_score_threshold:.2f}"] = f1
                
                # Average row
                if all_prec:
                    logger.info("  %s", separator)
                    avg_prec = np.mean(all_prec)
                    avg_rec = np.mean(all_rec)
                    avg_f1 = np.mean(all_f1)
                    
                    logger.info(
                        "  %s | %s | %s | %s",
                        "AVERAGE".ljust(rel_col_width),
                        f"{avg_prec:.3f}".center(metric_col_width),
                        f"{avg_rec:.3f}".center(metric_col_width),
                        f"{avg_f1:.3f}".center(metric_col_width)
                    )
            else:
                logger.info("  No relations found in predictions/annotations.")
        else:
            logger.info("  No relation annotations/predictions available.")
        
        logger.info("")

        if self.save_path_preds is not None:
            metrics_file = os.path.join(self.save_path_preds, "metrics.json")
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

        return metrics