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
        relation_score_thresholds (Sequence[float]): Relation score thresholds to sweep. Defaults to (0.3, 0.5, 0.7).
        distance_thresholds (Sequence[float]): Distance thresholds for recall@distance.
            Values in [0, 1] relative to max(img_w, img_h). Defaults to (0.001, 0.005, 0.01, 0.05).
        collect_device (str): Device for collecting results ('cpu' or 'gpu'). Defaults to 'cpu'.
        prefix (str, optional): Prefix for metric names. Defaults to None.
        save_path_preds (str, optional): Path to save prediction results. Defaults to None.
    """
    default_prefix: Optional[str] = 'keypointgraph'

    def __init__(self,
                 keypoint_score_threshold: float = 0.3,
                 keypoint_distance_threshold: float = 0.005,
                 keypoint_score_thresholds: Sequence[float] = (0.3, 0.5, 0.7),
                 relation_score_thresholds: Sequence[float] = (0.3, 0.5, 0.7),
                 distance_thresholds: Sequence[float] = (0.001, 0.005, 0.01, 0.05),
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 save_path_preds: str = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.distance_thresholds = tuple(distance_thresholds)
        self.keypoint_distance_threshold = float(keypoint_distance_threshold)
        self.keypoint_score_thresholds = tuple(keypoint_score_thresholds)
        self.keypoint_score_threshold = float(keypoint_score_threshold)
        self.relation_score_thresholds = tuple(relation_score_thresholds)
        self.save_path_preds = save_path_preds

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred = data_sample['pred_instances']
            img_w = int(data_sample['ori_shape'][1])
            img_h = int(data_sample['ori_shape'][0])
            max_dim = max(img_w, img_h)
            
            # Handle cropped images
            crop_bbox = data_sample.get('crop_bbox', None)
            gt_coords = data_sample['gt_instances']['keypoint_coords'].cpu().numpy().reshape(-1, 2)
            gt_labels = np.array([data_sample['text'][label] for label in data_sample['gt_instances']['labels']])
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
                gt_labels = gt_labels[gt_mask]
                
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
                gt_labels=gt_labels,
                gt_relations=gt_relations,
                relation_names=data_sample.get('relation_text', []),
                pred_coords=pred_coords,
                pred_labels=np.array(pred['label_names']),
                pred_scores=pred['scores'].cpu().numpy(),
                pred_relations=pred['relation_scores'].cpu().numpy(),
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
                    'keypoint_label_names': record['gt_labels'].tolist(),
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
        gt_labels_list = [r['gt_labels'] for r in results]
        pred_coords_list = [r['pred_coords'] for r in results]
        pred_labels_list = [r['pred_labels'] for r in results]
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
            gt_labels_list=gt_labels_list,
            pred_coords_list=pred_coords_list,
            pred_labels_list=pred_labels_list,
            pred_scores_list=pred_scores_list,
            max_dim_list=max_dim_list,
            distance_threshold_norm=self.keypoint_distance_threshold,
            score_threshold=score_thr_vec,
            distances_list=distances_list,
        )
        kp_rec = compute_recall(
            gt_coords_list=gt_coords_list,
            gt_labels_list=gt_labels_list,
            pred_coords_list=pred_coords_list,
            pred_labels_list=pred_labels_list,
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
            metrics[f"keypoints/prec@score{float(s):.2f}"] = float(p)
            metrics[f"keypoints/rec@score{float(s):.2f}"] = float(r_)
            metrics[f"keypoints/f1@score{float(s):.2f}"] = float(f)
        logger.info("")

        # 2) Keypoint recall vs distance thresholds at fixed keypoint score
        recalls_by_dist_arr = compute_recall(
            gt_coords_list=gt_coords_list,
            gt_labels_list=gt_labels_list,
            pred_coords_list=pred_coords_list,
            pred_labels_list=pred_labels_list,
            pred_scores_list=pred_scores_list,
            max_dim_list=max_dim_list,
            distance_threshold_norm=self.distance_thresholds,
            score_threshold=self.keypoint_score_threshold,
            distances_list=distances_list,
        )
        recalls_by_dist_arr = np.asarray(recalls_by_dist_arr, dtype=float)
        recall_by_distance = {t: float(recalls_by_dist_arr[i]) for i, t in enumerate(self.distance_thresholds)}

        logger.info("Keypoints: Recall@distance (score>=%.2f)", self.keypoint_score_threshold)
        for t in self.distance_thresholds:
            r_ = float(recall_by_distance.get(t, 0.0))
            logger.info("  dist<=%.4f -> Recall:%.3f", t, r_)
            metrics[f"keypoints/recall@dist{t}"] = r_
        logger.info("")

        # 3) Relation PR/F1 vs relation score thresholds at same fixed keypoint score and distance
        # Filter out images without relations or predictions
        if any(g is not None and p is not None for g, p in zip(gt_relations_list, pred_relations_list)):
            relation_curves = compute_precision_recall_relation(
                gt_coords_list=gt_coords_list,
                gt_labels_list=gt_labels_list,
                gt_relations_list=gt_relations_list,
                relation_names_list=relation_names_list,
                pred_coords_list=pred_coords_list,
                pred_labels_list=pred_labels_list,
                pred_scores_list=pred_scores_list,
                pred_relations_list=pred_relations_list,
                max_dim_list=max_dim_list,
                keypoint_score_threshold=self.keypoint_score_threshold,
                keypoint_distance_threshold=self.keypoint_distance_threshold,
                relation_score_thresholds=np.array(self.relation_score_thresholds, dtype=float),
                distances_list=distances_list
            )
            logger.info("Relations: Precision/Recall/F1 vs relation score thresholds (score>=%.2f, dist<=%.4f)", self.keypoint_score_threshold, self.keypoint_distance_threshold)
            for rel, curves in relation_curves.items():
                thrs = curves['thresholds']
                precs = curves['precision']
                recs = curves['recall']
                f1s = curves['f1']
                logger.info("  Relation: %s", rel)
                for s, p, r_, f in zip(thrs, precs, recs, f1s):
                    logger.info("    rel_score>=%.2f -> Precision:%.3f Recall:%.3f F1:%.3f", s, p, r_, f)
                    metrics[f"relations/{rel}/prec@score{float(s):.2f}"] = float(p)
                    metrics[f"relations/{rel}/rec@score{float(s):.2f}"] = float(r_)
                    metrics[f"relations/{rel}/f1@score{float(s):.2f}"] = float(f)
                logger.info("")
        else:
            logger.info("Relations: no relation annotations/predictions available.")
            logger.info("")

        if self.save_path_preds is not None:
            metrics_file = os.path.join(self.save_path_preds, "metrics.json")
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

        return metrics