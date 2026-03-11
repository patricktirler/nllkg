import numpy as np
from typing import Dict, List, Sequence, Any, Optional

from mmpose.evaluation.metrics.coco_metric import CocoMetric
from mmdet.registry import METRICS

from ...tools.graph_grouping import (
    group_keypoints_into_instances,
    make_is_valid_max_label,
)


@METRICS.register_module()
class PoseCocoMetric(CocoMetric):
    """COCO metric for pose estimation with keypoint grouping via graph merging.

    Args:
        ann_file (str, optional): Path to the coco format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to coco format. Defaults to None
        use_area (bool): Whether to use ``'area'`` message in the annotations.
            If the ground truth annotations (e.g. CrowdPose, AIC) do not have
            the field ``'area'``, please set ``use_area=False``.
            Defaults to ``True``
        iou_type (str): The same parameter as `iouType` in
            :class:`xtcocotools.COCOeval`, which can be ``'keypoints'``, or
            ``'keypoints_crowd'`` (used in CrowdPose dataset).
            Defaults to ``'keypoints'``
        score_mode (str): The mode to score the prediction results which
            should be one of the following options:

                - ``'bbox'``: Take the score of bbox as the score of the
                    prediction results.
                - ``'bbox_keypoint'``: Use keypoint score to rescore the
                    prediction results.
                - ``'bbox_rle'``: Use rle_score to rescore the
                    prediction results.
                - ``'keypoint'``: Use keypoint score to rescore the
                    prediction results.

            Defaults to ``'keypoint'``.
        keypoint_score_thr (float): The threshold of keypoint score. The
            keypoints with score lower than it will not be included to
            rescore the prediction results. Valid only when ``score_mode`` is
            ``bbox_keypoint``. Defaults to ``0.2``
        nms_mode (str): The mode to perform Non-Maximum Suppression (NMS),
            which should be one of the following options:

                - ``'oks_nms'``: Use Object Keypoint Similarity (OKS) to
                    perform NMS.
                - ``'soft_oks_nms'``: Use Object Keypoint Similarity (OKS)
                    to perform soft NMS.
                - ``'none'``: Do not perform NMS. Typically for bottomup mode
                    output.

            Defaults to ``'none'``.
        nms_thr (float): The Object Keypoint Similarity (OKS) threshold
            used in NMS when ``nms_mode`` is ``'oks_nms'`` or
            ``'soft_oks_nms'``. Will retain the prediction results with OKS
            lower than ``nms_thr``. Defaults to ``0.9``
        format_only (bool): Whether only format the output results without
            doing quantitative evaluation. This is designed for the need of
            test submission when the ground truth annotations are absent. If
            set to ``True``, ``outfile_prefix`` should specify the path to
            store the output results. Defaults to ``False``
        pred_converter (dict, optional): Config dictionary for the prediction
            converter. The dictionary has the same parameters as
            'KeypointConverter'. Defaults to None.
        gt_converter (dict, optional): Config dictionary for the ground truth
            converter. The dictionary has the same parameters as
            'KeypointConverter'. Defaults to None.
        outfile_prefix (str | None): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., ``'a/b/prefix'``.
            If not specified, a temp file will be created. Defaults to ``None``
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be ``'cpu'`` or
            ``'gpu'``. Defaults to ``'cpu'``
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, ``self.default_prefix``
            will be used instead. Defaults to ``None``
        node_score_thresh (float): Threshold for keypoint (node) scores. Default: 0.5
        edge_score_thresh (float): Minimum relation (edge) score to consider merging. Default: 0.5
        **kwargs: Other arguments passed to CocoMetric
    """

    def __init__(self,
                 ann_file: Optional[str] = None,
                 use_area: bool = True,
                 iou_type: str = 'keypoints',
                 score_mode: str = 'keypoint',
                 keypoint_score_thr: float = 0.2,
                 nms_mode: str = 'none',
                 nms_thr: float = 0.9,
                 format_only: bool = False,
                 pred_converter: Dict = None,
                 gt_converter: Dict = None,
                 outfile_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 node_score_thresh: float = 0.5,
                 edge_score_thresh: float = 0.5,
                 **kwargs):
        super().__init__(
            ann_file=ann_file,
            use_area=use_area,
            iou_type=iou_type,
            score_mode=score_mode,
            keypoint_score_thr=keypoint_score_thr,
            nms_mode=nms_mode,
            nms_thr=nms_thr,
            format_only=format_only,
            pred_converter=pred_converter,
            gt_converter=gt_converter,
            outfile_prefix=outfile_prefix,
            collect_device=collect_device,
            prefix=prefix
        )
        self.node_score_thresh = node_score_thresh
        self.edge_score_thresh = edge_score_thresh

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples: A batch of outputs from the model.
        """
        for data_sample in data_samples:
            if 'pred_instances' not in data_sample:
                raise ValueError(
                    '`pred_instances` are required to process the '
                    f'predictions results in {self.__class__.__name__}. ')

            # Extract raw predictions
            label_names = data_sample['pred_instances']['label_names']
            labels = np.array([self.dataset_meta['classes'].index(name) for name in label_names])
            scores = data_sample['pred_instances']['scores'].cpu().numpy()
            keypoints = data_sample['pred_instances']['keypoints'].cpu().numpy()  # (N,2)
            relation_scores = data_sample['pred_instances']['relation_scores'].cpu().numpy()  # (N,N,R or 1)

            # Filter nodes by score threshold
            valid_mask = scores >= self.node_score_thresh
            if np.any(valid_mask):
                valid_idx = np.nonzero(valid_mask)[0]
                f_labels = labels[valid_idx]
                f_scores = scores[valid_idx]
                f_kpts = keypoints[valid_idx]
                # Slice relation scores (preserve last dim R)
                R = relation_scores.shape[2]
                f_rel = relation_scores[np.ix_(valid_idx, valid_idx, np.arange(R))]
                # Group
                instance_groups = group_keypoints_into_instances(
                    keypoint_labels=f_labels,
                    keypoint_scores=f_scores,
                    relation_scores=f_rel,
                    is_valid=make_is_valid_max_label(1),
                    min_edge_score=self.edge_score_thresh
                )
                # Convert groups to COCO-format persons
                persons: List[Dict[str, Any]] = []
                num_kps = self.dataset_meta['num_keypoints']
                for grp in instance_groups:
                    kps_arr = np.zeros((num_kps, 3), dtype=np.float32)
                    for nid, lbl, sc in zip(grp.node_ids, grp.keypoint_labels, grp.keypoint_scores):
                        lbl_int = int(lbl)
                        if 0 <= lbl_int < num_kps:
                            coord = f_kpts[np.where(valid_idx == nid)[0][0]]  # map back within filtered set
                            kps_arr[lbl_int, 0] = float(coord[0])
                            kps_arr[lbl_int, 1] = float(coord[1])
                            kps_arr[lbl_int, 2] = float(sc)
                    if np.any(kps_arr[:, 2] > 0):
                        persons.append({
                            'image_id': data_sample['img_id'],
                            'category_id': 1,
                            'keypoints': kps_arr.reshape(-1).tolist(),
                            'score': float(np.mean(kps_arr[kps_arr[:, 2] > 0, 2]))
                        })
            else:
                persons = []

            # Build prediction dict (unchanged structure)
            if len(persons) == 0:
                pred = dict(
                    id=[],
                    img_id=data_sample['img_id'],
                    keypoints=np.zeros((0, self.dataset_meta['num_keypoints'], 2), dtype=np.float32),
                    keypoint_scores=np.zeros((0, self.dataset_meta['num_keypoints']), dtype=np.float32),
                    bbox_scores=np.zeros((0,), dtype=np.float32),
                    category_id=1
                )
            else:
                stacked_kpts = []
                stacked_scores = []
                for person in persons:
                    arr = np.array(person['keypoints']).reshape(-1, 3)
                    stacked_kpts.append(arr[:, :2])
                    stacked_scores.append(arr[:, 2])
                pred = dict(
                    id=[i for i in range(len(persons))],
                    img_id=data_sample['img_id'],
                    keypoints=np.stack(stacked_kpts, axis=0),
                    keypoint_scores=np.stack(stacked_scores, axis=0),
                    bbox_scores=np.ones(len(persons), dtype=np.float32),
                    category_id=data_sample['raw_ann_info'][0]['category_id']
                )

            # Ground truth packaging (unchanged)
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']
            if self.iou_type == 'keypoints_crowd':
                assert 'crowd_index' in data_sample, \
                    '`crowd_index` is required when `self.iou_type` is ' \
                    '`keypoints_crowd`'
                gt['crowd_index'] = data_sample['crowd_index']
            assert 'raw_ann_info' in data_sample, \
                'The row ground truth annotations are required for ' \
                'evaluation when `ann_file` is not provided'
            anns = data_sample['raw_ann_info']
            gt['raw_ann_info'] = anns if isinstance(anns, list) else [anns]

            self.results.append((pred, gt))