import os.path as osp
from typing import List, Union
import numpy as np
import os
import json
import copy

from mmdet.registry import DATASETS
from mmdet.datasets import CocoDataset

def generate_crop_coordinates(img_height, img_width, crop_height, crop_width, min_overlap=0):
    """Generate evenly spaced crop coordinates with minimum overlap.
    
    Args:
        img_height (int): Image height
        img_width (int): Image width
        crop_height (int): Crop height
        crop_width (int): Crop width
        min_overlap (float): Minimum overlap between crops (0-1). Defaults to 0.
        
    Returns:
        List[tuple]: List of (x1, y1, x2, y2) crop coordinates
    """
    crops = []
    
    # Calculate stride based on min_overlap
    stride_h = int(crop_height * (1 - min_overlap))
    stride_w = int(crop_width * (1 - min_overlap))
    
    # Ensure stride is at least 1
    stride_h = max(1, stride_h)
    stride_w = max(1, stride_w)
    
    # Calculate number of crops needed to cover the image
    num_crops_h = max(1, int(np.ceil((img_height - crop_height) / stride_h)) + 1)
    num_crops_w = max(1, int(np.ceil((img_width - crop_width) / stride_w)) + 1)
    
    # Generate evenly spaced positions
    if num_crops_h == 1:
        y_positions = [0]
    else:
        y_positions = [int(i * (img_height - crop_height) / (num_crops_h - 1)) 
                       for i in range(num_crops_h)]
    
    if num_crops_w == 1:
        x_positions = [0]
    else:
        x_positions = [int(i * (img_width - crop_width) / (num_crops_w - 1)) 
                       for i in range(num_crops_w)]
    
    # Generate all crop coordinates
    for y in y_positions:
        for x in x_positions:
            x1, y1 = x, y
            x2, y2 = x + crop_width, y + crop_height
            crops.append((x1, y1, x2, y2))
    
    return crops



@DATASETS.register_module()
class KeypointGraphDataset(CocoDataset):
    """
    
    Expected JSON structure:
    ```json
    {
        "images": [
            {
                "id": 1,
                "file_name": "image1.jpg",
                "height": 480,
                "width": 640
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "keypoints": [100, 200, 2, 300, 400, 2, ...],  // [x1, y1, v1, x2, y2, v2, ...]
                "keypoint_names": ["nose", "left_eye", "right_eye", ...],
                "keypoint_keypoint_relations": [
                    {
                        "name": "connected to",
                        "related_keypoints": {
                            "0": [1, 2],  // keypoint 0 is connected to keypoints 1 and 2
                            "1": [0],     // keypoint 1 is connected back to keypoint 0
                            ...
                        }
                    },
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "person",
                
            }
        ]
    }
    ```
    
    Notes:
    - Each annotation can define its own keypoint names and keypoint_relations
    - Multiple relation types are supported (e.g., "connected to", "part of")
    - Visibility values (v): 0=not labeled, 1=labeled but not visible, 2=labeled and visible
    
    Args:
        data_root (str): Root path of dataset
        ann_file (str): Annotation file path
        crop_sizes (List[tuple], optional): List of (height, width) tuples for crop sizes. If given, generates evenly spaced crops. Defaults to None.
        min_crop_overlap (float, optional): Minimum overlap between crops (0-1). Defaults to 0.
    """

    METAINFO = {}

    def __init__(self, data_root, ann_file, crop_sizes=None, min_crop_overlap=0, **kwargs):
        self.crop_sizes = crop_sizes
        self.min_crop_overlap = min_crop_overlap
        metainfo = self.load_metainfo(data_root, ann_file)
        super().__init__(data_root=data_root, ann_file=ann_file, metainfo=metainfo, **kwargs)

    def load_metainfo(self, data_root, ann_file: str) -> dict:
        if not os.path.isabs(ann_file):
            ann_file = osp.join(data_root, ann_file)

        # Load the annotation file
        with open(ann_file, 'r', encoding='utf-8') as f:
            raw_data_info = json.load(f)

        metainfo = {}
        
        unique_keypoint_names = set()
        unique_relation_names = set()
        annotations = raw_data_info.get('annotations', [])
        for ann in annotations:
            unique_keypoint_names.update(ann['keypoint_names'])
            if 'keypoint_relations' in ann:
                for relation in ann['keypoint_relations']:
                    unique_relation_names.add(relation['name'])
        metainfo['classes'] = tuple(sorted(unique_keypoint_names))
        metainfo['relation_names'] = tuple(sorted(unique_relation_names))

        return metainfo

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``
        Returns:
            List[dict]: A list of annotation.
        """
        base_data_list = super().load_data_list()
        
        if self.crop_sizes is None:
            return base_data_list
        
        # Apply cropping to each item in the base data list
        cropped_data_by_size = {crop_size: [] for crop_size in self.crop_sizes}
        
        for parsed_data_info in base_data_list:
            img_height = parsed_data_info['height']
            img_width = parsed_data_info['width']
            instances = parsed_data_info.get('instances', [])
            
            for crop_h_org, crop_w_org in self.crop_sizes:
                crop_h = min(crop_h_org, img_height)
                crop_w = min(crop_w_org, img_width)

                # Generate crop coordinates
                crop_coords = generate_crop_coordinates(
                    img_height, img_width, crop_h, crop_w, self.min_crop_overlap
                )
                
                # Create a copy of data_info for each crop, but only if it contains keypoints
                for crop_bbox in crop_coords:
                    x1, y1, x2, y2 = crop_bbox
                    
                    # Check if any keypoint is within this crop bbox
                    has_keypoints = False
                    for instance in instances:
                        kp_x, kp_y = instance['keypoint_coords']
                        if x1 <= kp_x < x2 and y1 <= kp_y < y2:
                            has_keypoints = True
                            break
                    
                    # Only add crop if it contains keypoints
                    if has_keypoints:
                        cropped_data = copy.deepcopy(parsed_data_info)
                        cropped_data['crop_bbox'] = crop_bbox  # (x1, y1, x2, y2)
                        cropped_data_by_size[(crop_h_org, crop_w_org)].append(cropped_data)
        
        # Concatenate all cropped data
        cropped_data_list = []
        for crop_size in self.crop_sizes:
            items = cropped_data_by_size[crop_size]
            cropped_data_list.extend(items)
        
        return cropped_data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']
        data_info = {}
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']
        data_info['custom_entities'] = True

        instances = []
        keypoint_id = 0

        # Collect all visible keypoint texts in the image
        visible_texts = set()
        for ann in ann_info:
            if ann.get('ignore', False):
                continue
            keypoint_texts = ann['keypoint_names']
            keypoints = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
            for (x, y, visible), text in zip(keypoints, keypoint_texts):
                if visible != 0:
                    visible_texts.add(text)
        visible_texts = sorted(visible_texts)
        data_info['text'] = tuple(visible_texts)
        text_to_label = {text: i for i, text in enumerate(visible_texts)}

        # Extract actual relation names used in this image's annotations
        actual_relation_names = set()
        for ann in ann_info:
            if ann.get('ignore', False) or 'keypoint_relations' not in ann:
                continue
            for relation in ann.get('keypoint_relations', []):
                actual_relation_names.add(relation.get('name'))
        relation_names = sorted(list(actual_relation_names))
        data_info['relation_text'] = tuple(relation_names)

        # First pass: create all keypoints and store keypoint index mappings
        keypoint_id_by_ann_and_idx = {}
        for ann in ann_info:
            if ann.get('ignore', False):
                continue
            keypoint_texts = ann['keypoint_names']
            keypoints = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
            for i, ((x, y, visible), text) in enumerate(zip(keypoints, keypoint_texts)):
                if visible == 0 or text not in text_to_label:
                    continue
                relation_dict = {rel_name: [] for rel_name in relation_names}
                instances.append({
                    'keypoint_id': keypoint_id,
                    'keypoint_coords': [x, y],
                    'keypoint_label': text_to_label[text],
                    'keypoint_name': text,
                    'ignore_flag': 0,
                    'keypoint_relations': relation_dict
                })
                keypoint_id_by_ann_and_idx[(ann['id'], i)] = keypoint_id
                keypoint_id += 1

        # Second pass: process keypoint_relations from annotations
        for ann in ann_info:
            if ann.get('ignore', False) or 'keypoint_relations' not in ann:
                continue
            ann_id = ann['id']
            for relation in ann.get('keypoint_relations', []):
                rel_name = relation.get('name')
                if rel_name not in relation_names:
                    continue
                # Modified structure: related_keypoints is a dict mapping src_idx (str) to list of dst_idx (int)
                related_keypoints = relation.get('related_keypoints', {})
                for src_str, dst_list in related_keypoints.items():
                    src_idx = int(src_str)
                    if (ann_id, src_idx) not in keypoint_id_by_ann_and_idx:
                        continue
                    src_id = keypoint_id_by_ann_and_idx[(ann_id, src_idx)]
                    src_instance = next((inst for inst in instances if inst['keypoint_id'] == src_id), None)
                    for dst_idx in dst_list:
                        if (ann_id, dst_idx) not in keypoint_id_by_ann_and_idx:
                            continue
                        dst_id = keypoint_id_by_ann_and_idx[(ann_id, dst_idx)]
                        dst_instance = next((inst for inst in instances if inst['keypoint_id'] == dst_id), None)
                        if src_instance and dst_instance:
                            src_instance['keypoint_relations'][rel_name].append(dst_id)

        data_info['raw_ann_info'] = ann_info
        data_info['instances'] = instances
        return data_info