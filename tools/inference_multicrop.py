import os
import json
import shutil
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from mmengine.config import Config

from nllkg.datasets.keypointgraph_dataset import generate_crop_coordinates
from nllkg.tools.inference import predinstances2dict, OpenVocPoseInferencer



def get_inferencer(
    cfg_path: str,
    weights: str,
    device: str = "cuda:0",
) -> Tuple["OpenVocPoseInferencer", Config]:
    """
    Instantiate the inferencer and parse the config.
 
    Parameters
    ----------
    cfg_path : str
        Path to configuration file.
    weights : str
        Path to model weights file.
    device : str, default="cuda:0"
        Device string passed to the inferencer.
 
    Returns
    -------
    tuple
        ``(inferencer, cfg)``.
    """
    inferencer = OpenVocPoseInferencer(model=cfg_path, weights=weights, device=device)
    cfg = Config.fromfile(cfg_path)
    return inferencer, cfg


def inference_multicrop_single(
    inferencer: "OpenVocPoseInferencer",
    cfg: Config,
    img_path: str,
    texts: str,
    relation_texts: str,
    temp_dir: str,
    keypoint_score_threshold: float = 0.3,
    batch_size: int = 1,
    progressive_crop_filtering: bool = False,
    save_dir: Optional[str] = None,
) -> Dict[str, list]:
    """
    Perform multi-scale crop-based inference on a single image.

    For each configured crop size this function:
    - Generates overlapping crops
    - Runs keypoint and relation inference on each crop
    - Translates crop-local keypoint coordinates into full-image space
    - Groups results by crop size

    Temporary crop images are written under `temp_dir` and removed after
    inference completes (or fails).

    Parameters
    ----------
    inferencer : OpenVocPoseInferencer
        Pre-initialised inferencer instance.
    cfg : Config
        Parsed mmengine config object (must contain ``crop_sizes`` and
        ``min_crop_overlap``).
    img_path : str
        Path to the input image file.
    texts : str
        Entity prompt text passed to the model.
    relation_texts : str
        Relation prompt text passed to the model.
    temp_dir : str
        Root directory for temporary crop storage.  A per-image sub-directory
        is created here and cleaned up on exit.
    keypoint_score_threshold : float, default=0.3
        Minimum score threshold for keypoint detections.
    batch_size : int, default=1
        Batch size for inference.
    progressive_crop_filtering : bool, default=False
        If True, sort crop sizes by area (largest first) and only process
        smaller crop sizes where keypoints were found in the larger one.
    save_dir : str, optional
        If provided, the result dict is written as a JSON file here.

    Returns
    -------
    dict
        ``crop_results_by_size`` mapping crop-size keys (e.g. ``"512x512"``)
        to lists of per-crop result dicts.
    """
    img_name = os.path.basename(img_path)
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    crop_sizes = cfg.crop_sizes
    if progressive_crop_filtering:
        crop_sizes = sorted(crop_sizes, key=lambda s: s[0] * s[1], reverse=True)

    img_temp_dir = os.path.join(temp_dir, img_name)
    os.makedirs(img_temp_dir, exist_ok=True)

    crop_results_by_size: Dict[str, list] = {}

    for crop_idx, crop_size in enumerate(crop_sizes):
            crop_key = f"{crop_size[1]}x{crop_size[0]}"
            crop_results_by_size[crop_key] = []

            crops = generate_crop_coordinates(
                h, w,
                crop_size[0], crop_size[1],
                cfg.min_crop_overlap,
            )

            # Progressive filtering: skip crop sizes with no overlap with
            # keypoints detected at the previous (larger) scale.
            if progressive_crop_filtering and crop_idx > 0:
                previous_crop_key = f"{crop_sizes[crop_idx - 1][1]}x{crop_sizes[crop_idx - 1][0]}"
                previous_keypoints = [
                    kp
                    for result in crop_results_by_size[previous_crop_key]
                    for kp in result["keypoint_coords"]
                ]

                if not previous_keypoints:
                    print(
                        f"Skipping crop size {crop_key} for {img_name} "
                        "(no keypoints from larger crops)"
                    )
                    continue

                prev_kp = np.asarray(previous_keypoints)
                crops = [
                    (x1, y1, x2, y2)
                    for x1, y1, x2, y2 in crops
                    if np.any(
                        (prev_kp[:, 0] >= x1) & (prev_kp[:, 0] <= x2) &
                        (prev_kp[:, 1] >= y1) & (prev_kp[:, 1] <= y2)
                    )
                ]

                if not crops:
                    print(
                        f"Skipping crop size {crop_key} for {img_name} "
                        "(no keypoints from larger crops overlap with this size)"
                    )
                    continue

            # Write crops to disk for batched inference.
            crop_folder = os.path.join(img_temp_dir, crop_key)
            os.makedirs(crop_folder, exist_ok=True)

            crop_paths: List[str] = []
            crop_bboxes: List[Tuple[int, int, int, int]] = []
            for ci, (x1, y1, x2, y2) in enumerate(crops):
                crop = img_rgb[y1:y2, x1:x2]
                crop_path = os.path.join(crop_folder, f"{ci:05d}.jpg")
                cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                crop_paths.append(crop_path)
                crop_bboxes.append((x1, y1, x2, y2))

            # Batched inference over all crops for this scale.
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

            for pred, (x1, y1, x2, y2) in zip(res["predictions"], crop_bboxes):
                rdict = predinstances2dict(pred.pred_instances)

                kp_coords = np.asarray(rdict.get("keypoint_coords", []))
                kp_scores = np.asarray(rdict.get("keypoint_scores", []))
                kp_labels = rdict.get("keypoint_label_names", [])
                rel_scores = np.asarray(rdict.get("relation_scores", []))

                if len(kp_coords) == 0:
                    continue

                keep = kp_scores >= keypoint_score_threshold
                kp_coords = kp_coords[keep]
                kp_scores = kp_scores[keep]
                kp_labels = [kp_labels[i] for i, k in enumerate(keep) if k]
                if rel_scores.size > 0:
                    rel_scores = rel_scores[np.ix_(keep, keep)]

                # Translate to full-image coordinates.
                kp_coords[:, 0] += x1
                kp_coords[:, 1] += y1

                crop_results_by_size[crop_key].append({
                    "keypoint_coords": kp_coords.tolist(),
                    "keypoint_scores": kp_scores.tolist(),
                    "keypoint_label_names": kp_labels,
                    "relation_label_names": [l.strip() for l in relation_texts.split(".")],
                    "relation_scores": rel_scores.tolist() if rel_scores.size > 0 else [],
                    "crop_bbox": (x1, y1, x2, y2),
                })

    shutil.rmtree(img_temp_dir, ignore_errors=True)

    if save_dir is not None:
        _save_image_inference_results(img_name, crop_results_by_size, save_dir)

    return crop_results_by_size


def inference_multicrop(
    cfg_path: str,
    weights: str,
    work_dir: str,
    img_dir: str,
    texts: str,
    relation_texts: str,
    keypoint_score_threshold: float = 0.3,
    batch_size: int = 1,
    progressive_crop_filtering: bool = False,
    save_dir: Optional[str] = None,
) -> Dict[str, Dict[str, list]]:
    """
    Perform multi-scale crop-based inference on all images in a directory.

    Iterates over every image in `img_dir`, delegates to
    :func:`inference_multicrop_single`, and collects the results.
    Temporary files for each image are cleaned up before moving to the next.

    Parameters
    ----------
    cfg_path : str
        Path to configuration file.
    weights : str
        Path to model weights file for inference.
    work_dir : str
        Working directory used for temporary crop storage.
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
        If True, sort crop sizes by area (largest first) and only perform
        inference on smaller crop sizes if keypoints were detected at the
        larger scale.
    save_dir : str, optional
        If provided, per-image JSON files are written here.

    Returns
    -------
    dict
        Mapping of ``img_name -> crop_results_by_size`` for every processed
        image.
    """
    temp_dir = os.path.join(work_dir, "_tmp_crops")
    os.makedirs(temp_dir, exist_ok=True)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    inferencer, cfg = get_inferencer(cfg_path=cfg_path, weights=weights, device="cuda:0")

    all_results: Dict[str, Dict[str, list]] = {}

    for img_name in sorted(os.listdir(img_dir)):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(img_dir, img_name)
        all_results[img_name] = inference_multicrop_single(
            inferencer=inferencer,
            cfg=cfg,
            img_path=img_path,
            texts=texts,
            relation_texts=relation_texts,
            temp_dir=temp_dir,
            keypoint_score_threshold=keypoint_score_threshold,
            batch_size=batch_size,
            progressive_crop_filtering=progressive_crop_filtering,
            save_dir=save_dir,
        )

    return all_results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_image_inference_results(
    img_name: str,
    crop_results_by_size: Dict[str, list],
    output_dir: str,
) -> str:
    """
    Persist per-image inference results to a JSON file.

    JSON structure::

        {
            "img_name": "...",
            "crop_results_by_size": {
                "512x512": [ ... ],
                "768x768": [ ... ]
            }
        }

    Parameters
    ----------
    img_name : str
        Original image filename (used to derive the JSON filename).
    crop_results_by_size : dict
        Inference results keyed by crop-size string.
    output_dir : str
        Directory in which the JSON file is written.

    Returns
    -------
    str
        Absolute path of the written JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}.json")
    with open(json_path, "w") as f:
        json.dump({"img_name": img_name, "crop_results_by_size": crop_results_by_size}, f, indent=2)
    return json_path