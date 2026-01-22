import numpy as np
import torch
import cv2
from mmcv.transforms import BaseTransform

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import get_box_type
from mmdet.datasets.transforms.formatting import PackDetInputs
from mmcv.transforms import to_tensor



@TRANSFORMS.register_module()
class LoadKeypointGraphAnnotationsAsBbox(BaseTransform):
    """Load and process the ``instances`` provided
    by dataset.

    Required Keys:

    - height
    - width
    - instances

      - keypoint_id: Unique ID for each keypoint instance
      - keypoint_coords: [x, y] coordinates of the keypoint
      - keypoint_label: Label index for the keypoint
      - ignore_flag: Whether to ignore this keypoint during training
      - keypoint_relations: Dictionary mapping relation names to lists of related keypoint IDs

    Added Keys:

    - gt_keypoint_coords (np.ndarray): Keypoint coordinates with shape (-1, 1, 2)
    - gt_bboxes (BaseBoxes[torch.float32]): Bounding boxes with shape (-1, 4)
    - gt_bboxes_labels (np.int64): Label indices for keypoints with shape (-1,)
    - gt_keypoint_ids (np.int64): Instance IDs for keypoints with shape (-1,)
    - gt_ignore_flags (bool): Ignore flags for keypoints with shape (-1,)
    - gt_keypoint_relations (dict): Dictionary mapping relation types to lists of [source_id, target_id] pairs

    Args:
        box_type (str): The box type used to wrap the bboxes. If ``box_type``
            is None, gt_bboxes will keep being np.ndarray. Defaults to 'hbox'.
    """

    def __init__(
            self,
            box_type: str = 'hbox',
            **kwargs) -> None:
        super().__init__(**kwargs)
        self.box_type = box_type

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation.
        """

        gt_bboxes = []
        gt_ignore_flags = []
        gt_keypoint_ids = []
        gt_keypoint_labels = []
        gt_keypoint_coords = []
        gt_keypoint_relations = {}
        
        instances = results.get('instances', [])
        
        gt_keypoint_relations = {}
        for instance in instances:
            bbox = [
                instance['keypoint_coords'][0],
                instance['keypoint_coords'][1],
                instance['keypoint_coords'][0],
                instance['keypoint_coords'][1]
            ]

            gt_bboxes.append(bbox)
            gt_ignore_flags.append(instance['ignore_flag'])
            gt_keypoint_ids.append(instance['keypoint_id'])
            gt_keypoint_coords.append([instance['keypoint_coords']])
            gt_keypoint_labels.append(instance['keypoint_label'])
            
            # Process keypoint_relations by type
            if 'keypoint_relations' in instance:
                for rel_type, related_ids in instance['keypoint_relations'].items():
                    if rel_type not in gt_keypoint_relations:
                        gt_keypoint_relations[rel_type] = []
                    for related_id in related_ids:
                        gt_keypoint_relations[rel_type].append([instance['keypoint_id'], related_id])

        if self.box_type is None:
            results['gt_bboxes'] = np.array(gt_bboxes, dtype=np.float32).reshape((-1, 4))
        else:
            _, box_type_cls = get_box_type(self.box_type)
            results['gt_bboxes'] = box_type_cls(gt_bboxes, dtype=torch.float32)

        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)
        results['gt_keypoint_ids'] = np.array(gt_keypoint_ids, dtype=np.int64)
        results['gt_keypoint_coords'] = np.array(gt_keypoint_coords, dtype=np.float32).reshape((-1, 1, 2))
        results['gt_keypoint_labels'] = np.array(gt_keypoint_labels, dtype=np.int64)
        results['gt_keypoint_relations'] = gt_keypoint_relations

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(box_size={self.box_size}, '
        repr_str += f'box_type={self.box_type})'
        return repr_str


@TRANSFORMS.register_module()
class ConvertRelationsToMatrix(BaseTransform):
    """Convert relation dictionary to adjacency matrices.
    
    Required Keys:
    - gt_keypoint_ids
    - gt_keypoint_relations (dict): Dictionary mapping relation types to lists of [source_id, target_id] pairs
    
    Added Keys:
    - gt_relation_matrices (ndarray): Adjacency matrices with shape (num_keypoints, num_keypoints, num_keypoint_relations)
    """
    
    def __init__(self, enforce_symmetry=True, **kwargs):
        super().__init__(**kwargs)
        self.enforce_symmetry = enforce_symmetry
    
    def transform(self, results: dict) -> dict:
        keypoint_ids = results['gt_keypoint_ids']
        keypoint_relations = results['gt_keypoint_relations']
        relation_types = list(keypoint_relations.keys())
        
        num_keypoints = len(keypoint_ids)
        num_keypoint_relations = len(relation_types)
        
        # Create a mapping from keypoint_id to index
        id_to_idx = {kid: idx for idx, kid in enumerate(keypoint_ids)}
        
        # Create a mapping from relation type to index
        rel_type_to_idx = {rel_type: idx for idx, rel_type in enumerate(relation_types)}
        
        # Initialize relation matrices with shape (num_keypoints, num_keypoints, num_keypoint_relations)
        relation_matrices = np.zeros((num_keypoints, num_keypoints, num_keypoint_relations), dtype=np.int64)
        
        # Fill the adjacency matrix for each relation type
        for rel_type, pairs in keypoint_relations.items():
            if rel_type not in rel_type_to_idx:
                continue
                
            rel_idx = rel_type_to_idx[rel_type]
            
            for src_id, dst_id in pairs:
                if src_id in id_to_idx and dst_id in id_to_idx:
                    src_idx = id_to_idx[src_id]
                    dst_idx = id_to_idx[dst_id]
                    relation_matrices[src_idx, dst_idx, rel_idx] = 1

        # Enforce symmetry if requested
        if self.enforce_symmetry:
            for rel_idx in range(num_keypoint_relations):
                mat = relation_matrices[:, :, rel_idx]
                relation_matrices[:, :, rel_idx] = np.logical_or(mat, mat.T).astype(np.int64)
        
        # Store relation matrices with shape (num_keypoints, num_keypoints, num_keypoint_relations)
        results['gt_relation_matrices'] = relation_matrices
        
        return results


@TRANSFORMS.register_module()
class TransformKeypoints(BaseTransform):
    def __init__(self, **kwargs):
        """Resize keypoints and set bboxes to a fixed size after transform.

        Required Keys:
        - homography_matrix
        - gt_keypoint_coords

        Modified Keys:
        - gt_keypoint_coords
        - gt_bboxes (if keypoint_box_size is not None)
        """
        super().__init__(**kwargs)

    def transform(self, results: dict) -> dict:
        homography_matrix = results.get('homography_matrix', None)

        if homography_matrix is None:
            return results

        gt_keypoint_coords = results.get('gt_keypoint_coords', None)
        if gt_keypoint_coords is not None:
            # gt_keypoint_coords shape: (N, 1, 2)
            N = gt_keypoint_coords.shape[0]
            keypoints_flat = gt_keypoint_coords.reshape(-1, 2)  # (N, 2)
            # Convert to homogeneous coordinates
            keypoints_h = np.concatenate([keypoints_flat, np.ones((N, 1), dtype=keypoints_flat.dtype)], axis=1)  # (N, 3)
            # Apply homography
            keypoints_transformed = (homography_matrix @ keypoints_h.T).T  # (N, 3)
            # Normalize
            keypoints_transformed = keypoints_transformed[:, :2] / keypoints_transformed[:, 2:3]
            results['gt_keypoint_coords'] = keypoints_transformed.reshape((-1, 1, 2))

        return results
    
    

@TRANSFORMS.register_module()
class PackKeypointGraphInputs(PackDetInputs):

    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_keypoint_labels': 'labels',
        'gt_keypoint_ids': 'keypoint_ids',
        'gt_keypoint_coords': 'keypoint_coords',
    }

    def transform(self, results: dict) -> dict:
        """
        Pack inputs and ignore keypoints that lie outside the image.
        """
        # ---- determine valid keypoints ----
        img_h, img_w = results['img_shape']

        keypoints = results.get('gt_keypoint_coords', None)
        if keypoints is not None:
            keypoints = np.asarray(keypoints)

            x = keypoints[:, 0, 0]
            y = keypoints[:, 0, 1]

            inside_mask = (
                (x >= 0) & (x < img_w) &
                (y >= 0) & (y < img_h)
            )
        else:
            inside_mask = None

        # combine with gt_ignore_flags if present
        if 'gt_ignore_flags' in results:
            ignore_mask = results['gt_ignore_flags'] == 0
            valid_mask = ignore_mask if inside_mask is None \
                else (ignore_mask & inside_mask)
        else:
            valid_mask = inside_mask

        # ---- filter fields BEFORE packing ----
        if valid_mask is not None:
            valid_idx = np.where(valid_mask)[0]

            for key in self.mapping_table.keys():
                if key not in results:
                    continue

                results[key] = results[key][valid_idx]

            # relation matrices: filter both axes
            if 'gt_relation_matrices' in results:
                rel = results['gt_relation_matrices']
                rel = rel[valid_idx][:, valid_idx, :]
                results['gt_relation_matrices'] = rel

            # update ignore flags to reflect filtering
            results['gt_ignore_flags'] = np.zeros(
                len(valid_idx), dtype=np.int64
            )

        # ---- call base packer ----
        packed_results = super().transform(results)

        # ---- attach relation matrices ----
        if 'gt_relation_matrices' in results:
            packed_results['data_samples'] \
                .gt_instances.relation_matrices = to_tensor(
                    results['gt_relation_matrices']
                )

        return packed_results



@TRANSFORMS.register_module()
class TopDownBBoxCrop(BaseTransform):
    """Crop image tightly to bbox using perspective warp.

    - Computes 4 bbox corners
    - Applies existing homography
    - Crops via perspective warp (rotation preserved)
    - Updates gt_bboxes by warping their corners
    - Optionally updates gt_keypoint_coords
    - Composes homography correctly
    """

    def __init__(self, transform_keypoints: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.transform_keypoints = transform_keypoints

    @staticmethod
    def _bbox_to_corners(bbox):
        """(x1,y1,x2,y2) -> (4,2) TL,TR,BR,BL"""
        x1, y1, x2, y2 = bbox
        return np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.float32,
        )

    @staticmethod
    def _apply_homography(points, H):
        """Apply homography to Nx2 points"""
        pts = np.concatenate(
            [points, np.ones((points.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        warped = (H @ pts.T).T
        return warped[:, :2] / warped[:, 2:3]

    @staticmethod
    def _perspective_crop(img, corners):
        """Perspective crop using warped bbox corners"""
        w1 = np.linalg.norm(corners[1] - corners[0])
        w2 = np.linalg.norm(corners[2] - corners[3])
        h1 = np.linalg.norm(corners[3] - corners[0])
        h2 = np.linalg.norm(corners[2] - corners[1])

        out_w = max(1, int(round(max(w1, w2))))
        out_h = max(1, int(round(max(h1, h2))))

        dst = np.array(
            [[0, 0],
             [out_w - 1, 0],
             [out_w - 1, out_h - 1],
             [0, out_h - 1]],
            dtype=np.float32,
        )

        H_crop = cv2.getPerspectiveTransform(
            corners.astype(np.float32), dst
        )
        crop = cv2.warpPerspective(img, H_crop, (out_w, out_h))
        return crop, H_crop

    @staticmethod
    def _warp_and_aabb(bboxes, H):
        """Warp bboxes by H and return axis-aligned boxes."""
        warped_boxes = []
        for b in bboxes:
            corners = TopDownBBoxCrop._bbox_to_corners(b)
            warped = TopDownBBoxCrop._apply_homography(corners, H)
            x1, y1 = warped.min(axis=0)
            x2, y2 = warped.max(axis=0)
            warped_boxes.append([x1, y1, x2, y2])
        return np.asarray(warped_boxes, dtype=np.float32)

    def transform(self, results: dict) -> dict:
        if 'bbox' not in results or 'img' not in results:
            return results

        img = results['img']
        bbox = np.asarray(results['bbox'], dtype=np.float32)

        H_prev = results.get(
            'homography_matrix', np.eye(3, dtype=np.float32)
        )

        # --- main bbox crop ---
        corners = self._bbox_to_corners(bbox)
        corners = self._apply_homography(corners, H_prev)
        crop, H_crop = self._perspective_crop(img, corners)

        # compose homography
        H_new = H_crop @ H_prev
        results['homography_matrix'] = H_new

        # --- update gt_bboxes ---
        if 'gt_bboxes' in results:
            gt = results['gt_bboxes']

            if hasattr(gt, 'tensor'):  # BaseBoxes
                b = gt.tensor.cpu().numpy()
                b = self._warp_and_aabb(b, H_new)
                b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, crop.shape[1])
                b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, crop.shape[0])
                gt.tensor = gt.tensor.new_tensor(b)

            else:  # numpy
                b = self._warp_and_aabb(gt, H_new)
                b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, crop.shape[1])
                b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, crop.shape[0])
                results['gt_bboxes'] = b

        # --- optionally update keypoints ---
        if self.transform_keypoints and 'gt_keypoint_coords' in results:
            kpts = results['gt_keypoint_coords']
            # (N,1,2) or (N,2)
            reshape_1d = False
            if kpts.ndim == 3:
                reshape_1d = True
                kpts = kpts[:, 0, :]
            kpts = self._apply_homography(kpts, H_new)
            if reshape_1d:
                kpts = kpts[:, None, :]
            results['gt_keypoint_coords'] = kpts

        # --- finalize ---
        results['img'] = crop
        results['img_shape'] = (crop.shape[0], crop.shape[1])
        results['bbox'] = np.array(
            [0, 0, crop.shape[1], crop.shape[0]],
            dtype=np.float32,
        )

        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(transform_keypoints={self.transform_keypoints})"

