import os
import json
from typing import Dict, List, Any

import cv2
import numpy as np
import shutil

from mmengine.config import Config


from nllkg.datasets.keypointgraph_dataset import generate_crop_coordinates
from nllkg.tools.inference import predinstances2dict, OpenVocPoseInferencer
from nllkg.tools.graph_grouping import group_keypoints_into_instances, InstanceGroup, IsValidFn
from nllkg.tools.graph_fitting import ShapeTemplate



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

def aggregate_keypoints_multicrop(
    inference_json_path: str,
    crop_size_key: str | None = None,
    keypoint_score_threshold: float = 0.3,
    dedup_distance_threshold: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate keypoints and relation scores across all crops of a given size
    into a single deduplicated set of global keypoints and a (N, N, R) relation matrix.

    For each pair of raw detections within `dedup_distance_threshold` pixels of
    each other (and sharing the same label), only the highest-scoring one is kept.
    Relation scores from all crops are accumulated into the global matrix by taking
    the element-wise maximum over all crops that observed a given pair.

    Parameters
    ----------
    inference_json_path : str
        Path to the per-image JSON file produced by `inference_multicrop`.
    crop_size_key : str, optional
        Crop size identifier. If None, the smallest available
        crop size is used.
    keypoint_score_threshold : float, default=0.3
        Minimum keypoint confidence score to include.
    dedup_distance_threshold : float, default=10.0
        Maximum pixel distance between two detections of the same label for them
        to be considered the same physical point.

    Returns
    -------
    keypoint_coords : np.ndarray, shape (N, 2)
        Deduplicated keypoint coordinates in full-image space.
    keypoint_scores : np.ndarray, shape (N,)
        Confidence scores for each deduplicated keypoint.
    keypoint_label_names : np.ndarray, shape (N,)
        Label name strings for each deduplicated keypoint.
    relation_scores : np.ndarray, shape (N, N, R)
        Aggregated relation scores. Entry [i, j, r] is the maximum relation score
        observed for the deduplicated pair (i, j) across all crops. Pairs that
        were never co-observed in any crop have a score of 0.
    relation_label_names : np.ndarray, shape (R,)
        Relation label name strings corresponding to the R relation channels.
    """
    with open(inference_json_path, "r") as f:
        data = json.load(f)

    crop_results_by_size = data["crop_results_by_size"]

    if crop_size_key is None:
        crop_size_key = min(
            crop_results_by_size.keys(),
            key=lambda k: eval(k.replace('x', '*'))
        )

    if crop_size_key not in crop_results_by_size:
        raise KeyError(
            f"Crop size '{crop_size_key}' not found. "
            f"Available: {list(crop_results_by_size.keys())}"
        )

    crop_results = crop_results_by_size[crop_size_key]

    # ------------------------------------------------------------------
    # Step 1: Pool all raw detections across crops
    # ------------------------------------------------------------------
    # Each raw detection keeps a back-reference to its source crop and its
    # local index within that crop — needed to look up relation_scores later.

    raw_coords  = []   # (M, 2)  full-image coords
    raw_scores  = []   # (M,)
    raw_labels  = []   # (M,)    str
    raw_crop_idx = []  # (M,)    which crop this came from
    raw_local_idx = [] # (M,)    index within that crop's arrays

    relation_label_names = None

    for ci, crop in enumerate(crop_results):
        coords = np.asarray(crop.get("keypoint_coords", []), dtype=float)
        scores = np.asarray(crop.get("keypoint_scores",  []), dtype=float)
        labels = crop.get("keypoint_label_names", [])

        if relation_label_names is None:
            relation_label_names = [l.strip() for l in crop.get("relation_label_names", [])]

        if len(coords) == 0:
            continue

        keep = scores >= keypoint_score_threshold
        coords = coords[keep]
        scores = scores[keep]
        labels = [labels[i] for i, k in enumerate(keep) if k]
        local_indices = np.where(keep)[0]

        for li, (coord, score, label) in enumerate(zip(coords, scores, labels)):
            raw_coords.append(coord)
            raw_scores.append(score)
            raw_labels.append(label)
            raw_crop_idx.append(ci)
            raw_local_idx.append(local_indices[li])

    if not raw_coords:
        R = len(relation_label_names) if relation_label_names else 0
        return (
            np.empty((0, 2), dtype=float),
            np.empty((0,),   dtype=float),
            np.empty((0,),   dtype=str),
            np.empty((0, 0, R), dtype=float),
            np.asarray(relation_label_names or []),
        )

    raw_coords    = np.asarray(raw_coords,    dtype=float)   # (M, 2)
    raw_scores    = np.asarray(raw_scores,    dtype=float)   # (M,)
    raw_labels    = np.asarray(raw_labels)                   # (M,)
    raw_crop_idx  = np.asarray(raw_crop_idx,  dtype=int)     # (M,)
    raw_local_idx = np.asarray(raw_local_idx, dtype=int)     # (M,)
    M = len(raw_coords)
    R = len(relation_label_names) if relation_label_names else 0

    # ------------------------------------------------------------------
    # Step 2: Greedy spatial deduplication
    # Process detections in descending score order so that when two
    # detections are duplicates, the higher-scoring one is the survivor.
    # ------------------------------------------------------------------

    # raw_to_dedup[m] = index of the deduplicated keypoint that raw detection m maps to.
    raw_to_dedup = np.full(M, -1, dtype=int)
    dedup_coords  = []
    dedup_scores  = []
    dedup_labels  = []

    order = np.argsort(-raw_scores)   # descending score

    for m in order:
        coord = raw_coords[m]
        label = raw_labels[m]

        # Check against already-accepted deduplicated points of the same label
        matched = -1
        for di, (dc, dl) in enumerate(zip(dedup_coords, dedup_labels)):
            if dl == label and np.linalg.norm(coord - dc) <= dedup_distance_threshold:
                matched = di
                break

        if matched == -1:
            # New unique detection
            matched = len(dedup_coords)
            dedup_coords.append(coord)
            dedup_scores.append(raw_scores[m])
            dedup_labels.append(label)

        raw_to_dedup[m] = matched

    N = len(dedup_coords)
    dedup_coords = np.asarray(dedup_coords, dtype=float)   # (N, 2)
    dedup_scores = np.asarray(dedup_scores, dtype=float)   # (N,)
    dedup_labels = np.asarray(dedup_labels)                # (N,)

    # ------------------------------------------------------------------
    # Step 3: Accumulate relation scores into the global (N, N, R) matrix
    # For each crop, map its local indices through raw_to_dedup and write
    # the crop's relation block into the global matrix, taking element-wise max.
    # ------------------------------------------------------------------

    global_rel = np.zeros((N, N, R), dtype=float)

    if R == 0:
        return dedup_coords, dedup_scores, dedup_labels, global_rel, np.asarray([])

    for ci, crop in enumerate(crop_results):
        crop_rel = np.asarray(crop.get("relation_scores", []), dtype=float)
        if crop_rel.ndim != 3 or crop_rel.shape[2] != R:
            continue

        # Find all raw detections that came from this crop
        # and survived the score threshold (i.e. have a dedup assignment)
        crop_mask = (raw_crop_idx == ci)
        if not np.any(crop_mask):
            continue

        local_indices  = raw_local_idx[crop_mask]   # local position in this crop's arrays
        dedup_indices  = raw_to_dedup[crop_mask]    # corresponding global dedup index

        # Validate local indices are within the crop's relation matrix
        valid = local_indices < crop_rel.shape[0]
        local_indices = local_indices[valid]
        dedup_indices = dedup_indices[valid]

        if len(local_indices) == 0:
            continue

        # Extract the relation sub-matrix for the detections we kept from this crop
        crop_rel_sub = crop_rel[np.ix_(local_indices, local_indices)]  # (k, k, R)

        # Scatter into the global matrix using the dedup indices
        global_rel[np.ix_(dedup_indices, dedup_indices)] = np.maximum(
            global_rel[np.ix_(dedup_indices, dedup_indices)],
            crop_rel_sub,
        )

    return (
        dedup_coords,
        dedup_scores,
        dedup_labels,
        global_rel,
        np.asarray(relation_label_names),
    )


def group_keypoints_multicrop(
    inference_json_path: str,
    is_valid_fn: IsValidFn,
    relation_label_names: List[str] = None,
    crop_size_key: str | None = None,
    keypoint_score_threshold: float = 0.3,
    relation_score_threshold: float = 0.3,
) -> List[List[InstanceGroup]]:
    """
    Group keypoints into instances for a single image and a selected crop size.

    This function loads a per-image inference JSON (containing results for multiple
    crop sizes), selects one crop size, filters keypoints by confidence, and groups
    them into instances using relation scores.

    By default, the largest available crop size is used.

    Parameters
    ----------
    inference_json_path : str
        Path to the per-image JSON file produced by crop-based inference.
    is_valid_fn: merged_group -> bool. Receives the prospective merged
        InstanceGroup. Returns True iff the merge should proceed.
        Passed directly to `group_keypoints_into_instances`.
    relation_label_names : List[str], optional
        List of relation label names to use during grouping. If None, all relation labels from the JSON are used.
    crop_size_key : str, optional
        Crop size identifier to use (e.g., "1024x1024").
        If None, the largest crop size (by area) is selected automatically.
    keypoint_score_threshold : float, default=0.3
        Minimum confidence score for keypoints to be considered.
    relation_score_threshold : float, default=0.3
        Minimum relation score for edges used during grouping.

    Returns
    -------
    List[List[InstanceGroup]]
        A list of grouped instances per crop for the selected crop size.
        Each element corresponds to one crop.
    """

    with open(inference_json_path, "r") as f:
        data = json.load(f)

    crop_results_by_size = data["crop_results_by_size"]

    # Select crop size
    if crop_size_key is None:
        # Choose the largest crop size by area
        def crop_area(k):
            w, h = map(int, k.split("x"))
            return w * h

        crop_size_key = max(crop_results_by_size.keys(), key=crop_area)

    if crop_size_key not in crop_results_by_size:
        raise KeyError(
            f"Crop size '{crop_size_key}' not found in {inference_json_path}. "
            f"Available sizes: {list(crop_results_by_size.keys())}"
        )

    crop_results = crop_results_by_size[crop_size_key]

    groups_per_crop = []

    for crop_result in crop_results:
        relation_scores = np.asarray(crop_result.get("relation_scores", []))
        relation_label_names_all = crop_result.get("relation_label_names", [])
        if relation_label_names is not None:
            relation_indices = [i for i, name in enumerate(relation_label_names_all) if name in relation_label_names]
            relation_scores = relation_scores[:, :, relation_indices]
        else:
            relation_label_names = relation_label_names_all

        groups = group_keypoints_into_instances(
            keypoint_scores=np.asarray(crop_result.get("keypoint_scores", [])),
            keypoint_coords=np.asarray(crop_result.get("keypoint_coords", [])),
            relation_scores=relation_scores,
            keypoint_label_names=crop_result.get("keypoint_label_names", []),
            relation_label_names=relation_label_names,
            is_valid=is_valid_fn,
            keypoint_score_threshold=keypoint_score_threshold,
            min_edge_score=relation_score_threshold,
        )

        groups_per_crop.append(groups)

    return groups_per_crop

def fit_shapes_multicrop(
    inference_json_path: str,
    templates: List[ShapeTemplate],
    crop_size_key: str | None = None,
    keypoint_score_threshold: float = 0.3,
    sigma: float = 10.0,
    method: str = "L-BFGS-B",
    **minimize_kwargs: Any,
) -> List[ShapeTemplate]:
    """
    Refine general shape templates using detected keypoints
    from a single-image inference result.

    This function:
    - Loads crop-based keypoint predictions from a per-image JSON file
    - Selects a crop size (smallest by default)
    - Aggregates keypoints across all crops of that size
    - Fits the provided shape templates

    Parameters
    ----------
    inference_json_path : str
        Path to the per-image inference JSON file.
    templates : List[ShapeTemplate]
        List of initial shape templates to be refined.
    crop_size_key : str, optional
        Crop size identifier to use (e.g., "1024x1024").
        If None, the smallest available crop size is used.
    keypoint_score_threshold : float, default=0.3
        Minimum keypoint confidence score to consider during fitting.
    sigma : float, default=10.0
        Gaussian bandwidth used in the fitting likelihood.
    method : str, default="L-BFGS-B"
        Optimization algorithm passed to `scipy.optimize.minimize`.
    **minimize_kwargs : dict
        Additional keyword arguments forwarded directly to `scipy.optimize.minimize`.

    Returns
    -------
    List[ShapeTemplate]
        List of refined shape templates after alignment to detected keypoints.
    """
    with open(inference_json_path, "r") as f:
        data = json.load(f)

    crop_results_by_size = data["crop_results_by_size"]

    if crop_size_key is None:
        crop_size_key = min(crop_results_by_size.keys(), key=lambda k: eval(k.replace('x', '*')))
    
    crop_results = crop_results_by_size[crop_size_key]

    all_coords, all_scores, all_labels = [], [], []
    for c in crop_results:
        coords = np.asarray(c.get("keypoint_coords", []))
        if len(coords) == 0: continue
        all_coords.append(coords)
        all_scores.append(np.asarray(c.get("keypoint_scores", [])))
        all_labels.extend([l.strip() for l in c.get("keypoint_label_names", [])])

    if not all_coords:
        return []
    
    keypoint_coords = np.concatenate(all_coords)
    keypoint_scores = np.concatenate(all_scores)
    keypoint_label_names = np.asarray(all_labels)
    
    mask = keypoint_scores >= keypoint_score_threshold
    keypoint_coords, keypoint_scores, keypoint_label_names = keypoint_coords[mask], keypoint_scores[mask], keypoint_label_names[mask]


    return [
        template.fit(
            keypoint_coords=keypoint_coords,
            keypoint_scores=keypoint_scores,
            keypoint_label_names=keypoint_label_names,
            sigma=sigma,
            method=method,
            **minimize_kwargs
        ) 
        for template in templates
    ]