import argparse
import json
import os
from typing import List, Dict, Any
from collections import defaultdict
import random

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load data from a JSON file and verify it's in COCO format."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict) or not all(k in data for k in ['images', 'annotations', 'categories']):
            raise ValueError(f"Not a valid COCO format dataset. Must contain 'images', 'annotations', and 'categories' keys.")
        return data
    except json.JSONDecodeError:
        raise ValueError(f"Error: {file_path} is not a valid JSON file.")
    except Exception as e:
        raise ValueError(f"Error loading {file_path}: {str(e)}.")

def get_stereo_key(item: Dict[str, Any], stereo_key_field: str = 'file_name') -> str:
    if stereo_key_field not in item:
        return str(item.get('id', ''))
    filename = item[stereo_key_field]
    base_name = filename.replace('left', '').replace('Left', '').replace('right', '').replace('Right', '')
    return base_name

def group_stereo_pairs(data: List[Dict[str, Any]], stereo_key_field: str = 'file_name') -> List[List[Dict[str, Any]]]:
    stereo_groups = defaultdict(list)
    for item in data:
        base_key = get_stereo_key(item, stereo_key_field)
        stereo_groups[base_key].append(item)
    return list(stereo_groups.values())

def split_coco_dataset(images: List[Dict[str, Any]], train_size: float, val_size: float, stereo: bool, stereo_key_field: str):
    if stereo:
        grouped_data = group_stereo_pairs(images, stereo_key_field)
        total = len(grouped_data)
        train_count = int(total * train_size)
        val_count = int(total * val_size)
        test_count = total - train_count - val_count
        # Shuffle for randomness
        random.seed(42)
        random.shuffle(grouped_data)
        train_groups = grouped_data[:train_count]
        val_groups = grouped_data[train_count:train_count + val_count]
        test_groups = grouped_data[train_count + val_count:]
        train_images = [item for group in train_groups for item in group]
        val_images = [item for group in val_groups for item in group]
        test_images = [item for group in test_groups for item in group]
    else:
        total = len(images)
        train_count = int(total * train_size)
        val_count = int(total * val_size)
        test_count = total - train_count - val_count
        random.seed(42)
        shuffled = images[:]
        random.shuffle(shuffled)
        train_images = shuffled[:train_count]
        val_images = shuffled[train_count:train_count + val_count]
        test_images = shuffled[train_count + val_count:]
    return train_images, val_images, test_images

def update_image_paths(images: List[Dict[str, Any]], split_name: str) -> None:
    """Update the file paths of images to point to the correct split directory."""
    for image in images:
        if 'path' in image:
            image['path'] = os.path.join("images", split_name, os.path.basename(image['path']))

def compute_statistics(images, annotations, categories):
    stats = {}
    stats['total_images'] = len(images)
    stats['total_annotations'] = len(annotations)
    stats['total_categories'] = len(categories)
    keypoint_counts = {}
    relation_counts = {}
    for ann in annotations:
        if 'keypoint_names' in ann:
            for kp in ann['keypoint_names']:
                keypoint_counts[kp] = keypoint_counts.get(kp, 0) + 1
        if 'keypoint_relations' in ann:
            for relation in ann['keypoint_relations']:
                if 'name' in relation:
                    name = relation['name']
                    relation_counts[name] = relation_counts.get(name, 0) + 1
    stats['num_keypoint_types'] = len(keypoint_counts)
    stats['keypoints'] = keypoint_counts
    stats['num_relation_types'] = len(relation_counts)
    stats['relations'] = relation_counts
    return stats

def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Save data to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data['images'])} images, {len(data['annotations'])} annotations to {file_path}")

def main():
    parser = argparse.ArgumentParser(description='Combine and split multiple COCO format JSON datasets')
    parser.add_argument('output_dir', help='Directory to save the combined datasets')
    parser.add_argument('json_files', nargs='+', help='COCO format JSON files to process')
    parser.add_argument('--train-size', type=float, default=0.7, help='Proportion of data for training set (default: 0.7)')
    parser.add_argument('--val-size', type=float, default=0.15, help='Proportion of data for validation set (default: 0.15)')
    parser.add_argument('--stereo', action='store_true', default=False, help='Treat data as stereo pairs that should stay together in splits')
    parser.add_argument('--stereo-key-field', type=str, default='file_name', help='Field to use for identifying stereo pairs (default: file_name)')
    args = parser.parse_args()

    test_size = 1 - args.train_size - args.val_size
    if test_size <= 0:
        print(f"Error: Invalid split proportions. Train ({args.train_size}) + Val ({args.val_size}) must be less than 1.0")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "annotations"), exist_ok=True)

    # Load and combine all datasets
    image_id_offset = 1
    annotation_id_offset = 1
    category_id_map = {}
    next_cat_id = 1

    # Store split results for each dataset
    split_images = {'train': [], 'val': [], 'test': []}
    split_annotations = {'train': [], 'val': [], 'test': []}
    split_categories = []

    for json_file in args.json_files:
        print(f"Processing {json_file}...")
        data = load_json_file(json_file)

        # Remap category IDs to avoid collisions
        local_cat_map = {}
        for cat in data['categories']:
            cat_tuple = tuple(sorted(cat.items()))
            if cat_tuple not in category_id_map:
                new_cat = dict(cat)
                new_cat['id'] = next_cat_id
                split_categories.append(new_cat)
                category_id_map[cat_tuple] = next_cat_id
                next_cat_id += 1
            local_cat_map[cat['id']] = category_id_map[cat_tuple]

        # Remap image and annotation IDs to avoid collisions
        local_image_map = {}
        local_images = []
        local_annotations = []
        for img in data['images']:
            new_img = dict(img)
            new_img['id'] = image_id_offset
            new_img['src_json'] = json_file
            local_image_map[img['id']] = image_id_offset
            local_images.append(new_img)
            image_id_offset += 1

        for ann in data['annotations']:
            new_ann = dict(ann)
            new_ann['id'] = annotation_id_offset
            new_ann['image_id'] = local_image_map[ann['image_id']]
            new_ann['category_id'] = local_cat_map[ann['category_id']]
            local_annotations.append(new_ann)
            annotation_id_offset += 1

        # Split images for this dataset
        train_imgs, val_imgs, test_imgs = split_coco_dataset(
            local_images,
            train_size=args.train_size,
            val_size=args.val_size,
            stereo=args.stereo,
            stereo_key_field=args.stereo_key_field
        )

        train_img_ids = {img['id'] for img in train_imgs}
        val_img_ids = {img['id'] for img in val_imgs}
        test_img_ids = {img['id'] for img in test_imgs}

        train_anns = [ann for ann in local_annotations if ann['image_id'] in train_img_ids]
        val_anns = [ann for ann in local_annotations if ann['image_id'] in val_img_ids]
        test_anns = [ann for ann in local_annotations if ann['image_id'] in test_img_ids]

        # Remove images with no annotation
        train_imgs = [img for img in train_imgs if any(ann['image_id'] == img['id'] for ann in train_anns)]
        val_imgs = [img for img in val_imgs if any(ann['image_id'] == img['id'] for ann in val_anns)]
        test_imgs = [img for img in test_imgs if any(ann['image_id'] == img['id'] for ann in test_anns)]

        split_images['train'].extend(train_imgs)
        split_images['val'].extend(val_imgs)
        split_images['test'].extend(test_imgs)
        split_annotations['train'].extend(train_anns)
        split_annotations['val'].extend(val_anns)
        split_annotations['test'].extend(test_anns)

    # Use all_categories from all datasets
    train_imgs = split_images['train']
    val_imgs = split_images['val']
    test_imgs = split_images['test']
    train_anns = split_annotations['train']
    val_anns = split_annotations['val']
    test_anns = split_annotations['test']

    update_image_paths(train_imgs, "train")
    update_image_paths(val_imgs, "val")
    update_image_paths(test_imgs, "test")

    train_data = {
        'statistics': compute_statistics(train_imgs, train_anns, split_categories),
        'images': train_imgs,
        'annotations': train_anns,
        'categories': split_categories
    }
    val_data = {
        'statistics': compute_statistics(val_imgs, val_anns, split_categories),
        'images': val_imgs,
        'annotations': val_anns,
        'categories': split_categories
    }
    test_data = {
        'statistics': compute_statistics(test_imgs, test_anns, split_categories),
        'images': test_imgs,
        'annotations': test_anns,
        'categories': split_categories
    }

    save_json(train_data, os.path.join(args.output_dir, "annotations", "train.json"))
    save_json(val_data, os.path.join(args.output_dir, "annotations", "val.json"))
    save_json(test_data, os.path.join(args.output_dir, "annotations", "test.json"))

if __name__ == "__main__":
    main()