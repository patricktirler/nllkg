import os
import json
from typing import Dict
import cv2
import numpy as np
import shutil

from mmengine.config import Config

from nllkg.datasets.keypointgraph_dataset import generate_crop_coordinates
from nllkg.tools.inference import predinstances2dict, OpenVocPoseInferencer


def inference_multicrop(
    cfg_path,
    weights: str,
    work_dir: str,
    save_dir: str,
    img_dir: str,
    texts: str,
    relation_texts: str,
    keypoint_score_threshold: float = 0.3,
    batch_size: int = 1,
    progressive_crop_filtering: bool = False,
):
    """
    Perform multi-scale crop-based inference on images and save a single JSON per image.

    For each image in `img_dir`, this function:
    - Generates overlapping crops at multiple configured crop sizes
    - Runs keypoint and relation inference on each crop
    - Translates crop-local keypoint coordinates into full-image space
    - Groups results by crop size
    - Saves all results into one JSON file per image

    Temporary crop images are written to disk during inference and removed afterward.

    Parameters
    ----------
    cfg_path : str
        Path to configuration file.
    weights : str
        Path to model weights file for inference.
    work_dir : str
        Working directory used for temporary crop storage.
    save_dir : str
        Directory where per-image JSON files are written.
    img_dir : str
        Directory containing input images.
    texts : str
        Entity prompt text passed to the model.
    relation_texts : str
        Relation prompt text passed to the model.
    keypoint_score_threshold : float, default=0.3
        Minimum score threshold for keypoint detections.
    batch_size : int, default=1
        Batch size for inference.
    progressive_crop_filtering : bool, default=False
        If True, sort crop sizes by area (largest first) and only perform inference
        on smaller crop sizes if keypoints were detected in the larger crop size.
    """
    temp_dir = os.path.join(work_dir, "_tmp_crops")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    inferencer = OpenVocPoseInferencer(
        model=cfg_path,
        weights=weights,
        device='cuda:0',
    )
    cfg = Config.fromfile(cfg_path)

    crop_sizes = cfg.crop_sizes
    if progressive_crop_filtering:
        crop_sizes = sorted(
            crop_sizes,
            key=lambda s: s[0] * s[1],
            reverse=True,
        )

    try:
        for img_name in sorted(os.listdir(img_dir)):
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]

        
            crop_results_by_size = {}
            for crop_idx, crop_size in enumerate(crop_sizes):
                crop_key = f"{crop_size[1]}x{crop_size[0]}"
                crop_results_by_size[crop_key] = []

                crops = generate_crop_coordinates(
                    h, w,
                    crop_size[0], crop_size[1],
                    cfg.min_crop_overlap,
                )

                # If progressive filtering is enabled and we're not on the first crop size,
                # filter crops to only those that have keypoints from the previous crop size
                if progressive_crop_filtering and crop_idx > 0:
                    # Gather keypoints from previous crop size results
                    previous_crop_key = f"{crop_sizes[crop_idx - 1][1]}x{crop_sizes[crop_idx - 1][0]}"
                    previous_keypoints = []
                    for result in crop_results_by_size[previous_crop_key]:
                        previous_keypoints.extend(result["keypoint_coords"])
                    
                    if previous_keypoints:
                        previous_keypoints = np.asarray(previous_keypoints)
                        
                        # Vectorized check: for each crop, check if any keypoint falls within
                        filtered_crops = []
                        for x1, y1, x2, y2 in crops:
                            if np.any(
                                (previous_keypoints[:, 0] >= x1) &
                                (previous_keypoints[:, 0] <= x2) &
                                (previous_keypoints[:, 1] >= y1) &
                                (previous_keypoints[:, 1] <= y2)
                            ):
                                filtered_crops.append((x1, y1, x2, y2))
                        
                        if not filtered_crops:
                            print(
                                f"Skipping crop size {crop_key} for {img_name} "
                                "(no keypoints from larger crops overlap with this size)"
                            )
                            continue
                        
                        crops = filtered_crops
                    else:
                        print(
                            f"Skipping crop size {crop_key} for {img_name} "
                            "(no keypoints from larger crops)"
                        )
                        continue

                # Folder for this crop size
                crop_folder = os.path.join(
                    temp_dir, img_name, crop_key
                )
                os.makedirs(crop_folder, exist_ok=True)

                crop_paths = []
                crop_bboxes = []

                # Save all crops first
                for ci, (x1, y1, x2, y2) in enumerate(crops):
                    crop = img_rgb[y1:y2, x1:x2]

                    crop_path = os.path.join(
                        crop_folder,
                        f"{ci:05d}.jpg"
                    )
                    cv2.imwrite(
                        crop_path,
                        cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                    )

                    crop_paths.append(crop_path)
                    crop_bboxes.append((x1, y1, x2, y2))

                # ------------------------------------------------------
                # SINGLE BATCHED INFERENCE CALL
                # ------------------------------------------------------
                res = inferencer(
                    crop_folder,
                    texts=texts,
                    relation_texts=relation_texts,
                    batch_size=batch_size,
                    return_datasamples=True,
                    no_save_vis=True,
                    no_save_pred=True,
                    pred_score_thr=keypoint_score_threshold,
                    custom_entities=True,
                )

                # ------------------------------------------------------
                # PROCESS RESULTS
                # ------------------------------------------------------
                predictions = res["predictions"]

                for pred, (x1, y1, x2, y2) in zip(predictions, crop_bboxes):
                    rdict = predinstances2dict(pred.pred_instances)

                    kp_coords = np.asarray(rdict.get("keypoint_coords", []))
                    kp_scores = np.asarray(rdict.get("keypoint_scores", []))
                    kp_labels = rdict.get("keypoint_label_names", [])
                    rel_scores = np.asarray(rdict.get("relation_scores", []))

                    if len(kp_coords) == 0:
                        continue

                    # Thresholding
                    keep = kp_scores >= keypoint_score_threshold
                    kp_coords = kp_coords[keep]
                    kp_scores = kp_scores[keep]
                    kp_labels = [kp_labels[i] for i, k in enumerate(keep) if k]

                    if rel_scores.size > 0:
                        rel_scores = rel_scores[np.ix_(keep, keep)]

                    # Translate to full-image coordinates
                    kp_coords[:, 0] += x1
                    kp_coords[:, 1] += y1

                    crop_results_by_size[crop_key].append({
                        "keypoint_coords": kp_coords.tolist(),
                        "keypoint_scores": kp_scores.tolist(),
                        "keypoint_label_names": kp_labels,
                        "relation_label_names": [l.strip() for l in relation_texts.split('.')],
                        "relation_scores": (
                            rel_scores.tolist() if rel_scores.size > 0 else []
                        ),
                        "crop_bbox": (x1, y1, x2, y2),
                    })

            # Save final JSONs
            save_image_inference_results(
                img_name,
                crop_results_by_size,
                save_dir,
            )

            print(f"Saved inference JSON for {img_name}")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

def save_image_inference_results(
    img_name: str,
    crop_results_by_size: Dict[str, list],
    output_dir: str,
) -> str:
    """
    Save inference results for a single image across all crop sizes.

    JSON structure:
    {
        "img_name": "...",
        "crop_results_by_size": {
            "512x512": [ ... ],
            "768x768": [ ... ]
        }
    }
    """

    json_path = os.path.join(
        output_dir,
        f"{img_name[:-4]}.json"
    )

    json_data = {
        "img_name": img_name,
        "crop_results_by_size": crop_results_by_size,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    return json_path

