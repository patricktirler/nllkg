import os
import json
from typing import Dict, List, Tuple

import cv2
import numpy as np
import shutil

from mmengine.config import Config
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches


from nllkp.datasets.keypointgraph_dataset import generate_crop_coordinates
from nllkp.tools.inference import predinstances2dict, OpenVocPoseInferencer
from nllkp.tools.graph_grouping import group_keypoints_into_instances, InstanceGroup, CheckMergeFn
from nllkp.tools.graph_fitting import ShapeTemplate, fit_shapes_with_keypoints



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
                    rel_scores = np.asarray(
                        rdict.get("keypoint_relation_scores", [])
                    )

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
                        "keypoint_relation_scores": (
                            rel_scores.tolist() if rel_scores.size > 0 else []
                        ),
                        "crop_bbox": (x1, y1, x2, y2),
                    })

            # Save final JSON
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


def group_keypoints_multicrop(
    inference_json_path: str,
    check_merge_fn: CheckMergeFn,
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
    check_merge_fn : callable
        Function used to determine whether two keypoints or groups may be merged.
        Passed directly to `group_keypoints_into_instances`.
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
        groups = group_keypoints_into_instances(
            keypoint_scores=np.asarray(
                crop_result.get("keypoint_scores", [])
            ),
            keypoint_coords=np.asarray(
                crop_result.get("keypoint_coords", [])
            ),
            relation_scores=np.asarray(
                crop_result.get("keypoint_relation_scores", [])
            ),
            keypoint_label_names=crop_result.get(
                "keypoint_label_names", []
            ),
            check_merge=check_merge_fn,
            keypoint_score_threshold=keypoint_score_threshold,
            min_edge_score=relation_score_threshold,
        )

        groups_per_crop.append(groups)

    return groups_per_crop


def fit_shapes_multicrop(
    inference_json_path: str,
    templates: List[ShapeTemplate],
    dof: str = "trsxsy",
    crop_size_key: str | None = None,
    keypoint_score_threshold: float = 0.3,
    sigma: float = 10.0,
) -> List[ShapeTemplate]:
    """
    Refine general shape templates using detected keypoints
    from a single-image inference result.

    This function:
    - Loads crop-based keypoint predictions from a per-image JSON file
    - Selects a crop size (smallest by default)
    - Aggregates keypoints across all crops of that size
    - Fits the provided shape templates to the observed keypoints using
      a parametric transformation model

    Parameters
    ----------
    inference_json_path : str
        Path to the per-image inference JSON file.
    templates : List[ShapeTemplate]
        List of initial shape templates to be refined.
    dof : str, default="trs"
        Degrees of freedom for template fitting:
        - "t": translation only
        - "tr": translation + rotation
        - "trs": translation + rotation + uniform scale
        - "trsxsy": translation + rotation + independent x/y scale
        - "perspective": affine / projective transform
    crop_size_key : str, optional
        Crop size identifier to use (e.g., "1024x1024").
        If None, the smallest available crop size is used.
    keypoint_score_threshold : float, default=0.3
        Minimum keypoint confidence score to consider during fitting.
    sigma : float, default=10.0
        Gaussian bandwidth used in the fitting likelihood.

    Returns
    -------
    List[ShapeTemplate]
        List of refined shape templates after alignment to detected keypoints.
    """

    # Load inference JSON
    with open(inference_json_path, "r") as f:
        data = json.load(f)

    crop_results_by_size = data["crop_results_by_size"]

    # Select crop size (default: smallest by area)
    if crop_size_key is None:
        def crop_area(k):
            w, h = map(int, k.split("x"))
            return w * h

        crop_size_key = min(crop_results_by_size.keys(), key=crop_area)

    if crop_size_key not in crop_results_by_size:
        raise KeyError(
            f"Crop size '{crop_size_key}' not found in inference results. "
            f"Available sizes: {list(crop_results_by_size.keys())}"
        )
    
    crop_results = crop_results_by_size[crop_size_key]

    # Aggregate keypoints across all crops
    all_coords = []
    all_scores = []
    all_label_names = []

    for c in crop_results:
        coords = np.asarray(c.get("keypoint_coords", []))
        scores = np.asarray(c.get("keypoint_scores", []))
        label_names = [l.strip() for l in c.get("keypoint_label_names", [])]

        if len(coords) == 0:
            continue

        all_coords.append(coords)
        all_scores.append(scores)
        all_label_names.extend(label_names)

    if not all_coords:
        return []

    kp_coords = np.concatenate(all_coords, axis=0)
    kp_scores = np.concatenate(all_scores, axis=0)
    kp_label_names = np.asarray(all_label_names)

    # Refine templates using keypoint-based fitting
    refined_templates = fit_shapes_with_keypoints(
        keypoint_scores=kp_scores,
        keypoint_coords=kp_coords,
        keypoint_label_names=kp_label_names,
        template_keypoints=templates,
        dof=dof,
        keypoint_score_threshold=keypoint_score_threshold,
        sigma=sigma,
    )

    return refined_templates

def visualize_multicrop(
    img_path: str,
    inference_json_path: str,
    instance_groups: List[List['InstanceGroup']],
    fits: List['ShapeTemplate'],
    initial_guesses: List['ShapeTemplate'],
    save_dir: str,
    keypoint_score_threshold: float = 0.3,
    relation_score_threshold: float = 0.1,
) -> None:

    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Load image
    # ------------------------------------------------------------
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")

    H, W = img_bgr.shape[:2]

    # ------------------------------------------------------------
    # Load inference JSON
    # ------------------------------------------------------------
    with open(inference_json_path, "r") as f:
        data = json.load(f)

    crop_sizes = data.get("crop_results_by_size", {})
    if not crop_sizes:
        raise ValueError(f"No crop results found in {inference_json_path}")

    def crop_area(k):
        w, h = map(int, k.split("x"))
        return w * h

    largest_key = max(crop_sizes.keys(), key=crop_area)
    smallest_key = min(crop_sizes.keys(), key=crop_area)

    largest_crop_results = crop_sizes[largest_key]
    smallest_crop_results = crop_sizes[smallest_key]

    # ------------------------------------------------------------
    # Collect all label names for consistent coloring
    # ------------------------------------------------------------
    all_labels = set()

    for crop_list in [largest_crop_results, smallest_crop_results]:
        for crop in crop_list:
            all_labels.update(crop.get("keypoint_label_names", []))

    for crop_groups in instance_groups:
        for group in crop_groups:
            labels = getattr(group, "keypoint_label_names", None)
            if labels is not None:
                all_labels.update(labels)

    for template in fits + initial_guesses:
        if template.keypoint_label_names is not None:
            all_labels.update(template.keypoint_label_names)

    all_labels = sorted(list(all_labels))
    label_to_color = _get_label_colors(all_labels)

    # Convert BGR to RGB for legend
    label_to_rgb = {
        k: tuple(reversed(v)) for k, v in label_to_color.items()
    }

    REL_COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]

    # ============================================================
    # 1️⃣ Largest crop inference
    # ============================================================
    vis_crops = img_bgr.copy()

    # Draw crop boxes
    bboxes = [
        tuple(map(int, crop["crop_bbox"]))
        for crop in largest_crop_results
        if "crop_bbox" in crop
    ]
    if bboxes:
        vis_crops = _draw_crops_with_borders(vis_crops, bboxes)

    for crop in largest_crop_results:
        kp_coords = np.asarray(crop.get("keypoint_coords", []))
        kp_scores = np.asarray(crop.get("keypoint_scores", []))
        kp_labels = crop.get("keypoint_label_names", [])

        keep = kp_scores >= keypoint_score_threshold
        kp_coords = kp_coords[keep]
        kp_labels = [kp_labels[i] for i, k in enumerate(keep) if k]

        if len(kp_coords) == 0:
            continue

        relations = np.asarray(crop.get("keypoint_relation_scores", []))
        if relations.size > 0:
            relations = relations[np.ix_(keep, keep)]
            N, _, R = relations.shape
            for i in range(N):
                for j in range(i + 1, N):
                    p1 = tuple(kp_coords[i].astype(int))
                    p2 = tuple(kp_coords[j].astype(int))
                    for r in range(R):
                        if relations[i, j, r] >= relation_score_threshold:
                            cv2.line(
                                vis_crops,
                                p1, p2,
                                REL_COLORS[r % len(REL_COLORS)],
                                2
                            )

        vis_crops = _draw_keypoints_cv2(
            vis_crops, kp_coords, kp_labels, label_to_color
        )

    # ============================================================
    # 2️⃣ Instance groups + initial guesses
    # ============================================================
    vis_groups = img_bgr.copy()
    if bboxes:
        vis_groups = _draw_crops_with_borders(vis_groups, bboxes)

    for crop_groups in instance_groups:
        for group in crop_groups:
            coords = np.asarray(group.keypoint_coords)
            labels = getattr(group, "keypoint_label_names", None)

            if coords is None or len(coords) == 0:
                continue

            if labels is None:
                labels = [None] * len(coords)

            # Draw connections from adjacency matrix
            K, _, R = group.adjacency_matrix.shape
            for i in range(K):
                for j in range(i + 1, K):
                    p1 = tuple(coords[i].astype(int))
                    p2 = tuple(coords[j].astype(int))
                    for r in range(R):
                        score = group.adjacency_matrix[i, j, r]
                        if score >= relation_score_threshold:
                            color = REL_COLORS[r % len(REL_COLORS)]
                            cv2.line(vis_groups, p1, p2, color, int(0.5 + 2 * score))

            vis_groups = _draw_keypoints_cv2(
                vis_groups, coords, labels, label_to_color
            )

    for template in initial_guesses:
        _draw_shape_template(vis_groups, template, color=(0, 255, 255))

    # ============================================================
    # 3️⃣ Smallest crop + fitted shapes
    # ============================================================
    vis_fits = img_bgr.copy()

    small_bboxes = [
        tuple(map(int, crop["crop_bbox"]))
        for crop in smallest_crop_results
        if "crop_bbox" in crop
    ]
    if small_bboxes:
        vis_fits = _draw_crops_with_borders(vis_fits, small_bboxes)

    for crop in smallest_crop_results:
        kp_coords = np.asarray(crop.get("keypoint_coords", []))
        kp_scores = np.asarray(crop.get("keypoint_scores", []))
        kp_labels = crop.get("keypoint_label_names", [])

        keep = kp_scores >= keypoint_score_threshold
        kp_coords = kp_coords[keep]
        kp_labels = [kp_labels[i] for i, k in enumerate(keep) if k]

        if len(kp_coords) > 0:
            vis_fits = _draw_keypoints_cv2(
                vis_fits, kp_coords, kp_labels, label_to_color
            )

    for template in fits:
        _draw_shape_template(vis_fits, template, color=(255, 0, 0))

    # ============================================================
    # Dynamic figure size based on image resolution
    # ============================================================
    fig, axs = plt.subplots(3, 1, figsize=(20, 20))

    axs[0].imshow(cv2.cvtColor(vis_crops, cv2.COLOR_BGR2RGB))
    axs[0].set_title(f"1. Largest Crop Inference ({largest_key})")
    axs[0].axis("off")

    axs[1].imshow(cv2.cvtColor(vis_groups, cv2.COLOR_BGR2RGB))
    axs[1].set_title("2. Instance Groups + Initial Guesses")
    axs[1].axis("off")

    axs[2].imshow(cv2.cvtColor(vis_fits, cv2.COLOR_BGR2RGB))
    axs[2].set_title(f"3. Fitted Shapes ({smallest_key})")
    axs[2].axis("off")

    # ------------------------------------------------------------
    # Legend (first subplot)
    # ------------------------------------------------------------

    legend_handles = [
        mpatches.Patch(
            color=np.array(label_to_rgb[label]) / 255.0,
            label=label
        )
        for label in all_labels
    ]

    axs[0].legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
        ncol=1
    )

    plt.tight_layout()

    out_path = os.path.join(save_dir, os.path.basename(img_path))
    plt.savefig(out_path, dpi=500, bbox_inches="tight")
    plt.close(fig)


def _get_label_colors(labels: List[str]) -> dict:
    n = len(labels)
    colormap = cm.get_cmap("rainbow", n)
    label_to_color = {}
    for i, label in enumerate(labels):
        rgb = colormap(i)[:3]  # float RGB [0,1]
        bgr = tuple(int(255*c) for c in rgb[::-1])  # BGR
        label_to_color[label] = bgr
    return label_to_color


def _draw_keypoints_cv2(img: np.ndarray, k_coords: np.ndarray,
                       k_labels: List[str],
                       label_to_color: dict,
                       radius: int = 5) -> np.ndarray:
    vis = img.copy()
    for coord, label in zip(k_coords, k_labels):
        color = label_to_color.get(label, (0, 255, 0))
        x, y = int(coord[0]), int(coord[1])
        cv2.circle(vis, (x, y), radius, color, -1)
        cv2.circle(vis, (x, y), radius, (255, 255, 255), 1)
    return vis


def _draw_shape_template(img: np.ndarray, template, color=(0, 255, 255)) -> None:
    coords = template.keypoint_coords
    if len(coords) == 0:
        return
    pts = coords.astype(int)
    if len(pts) > 2:
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
    for pt in pts:
        cv2.circle(img, tuple(pt), 4, color, -1)
        cv2.circle(img, tuple(pt), 2, (255, 255, 255), 1)


def _draw_crops_with_borders(
    img: np.ndarray,
    crops: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (200, 200, 200),
    thickness: int = 2,
    corner_length_ratio: float = 0.04,
    corner_thickness: int = 8,
) -> np.ndarray:
    """
    Draw crop rectangles with thick L-shaped corner markers.

    Parameters
    ----------
    img : np.ndarray
        Image in BGR format.
    crops : list of (x1, y1, x2, y2)
        Crop bounding boxes.
    color : tuple
        BGR color.
    thickness : int
        Border thickness.
    corner_length_ratio : float
        Length of corner markers as fraction of box size.
    corner_thickness : int
        Thickness of corner marker lines.
    """

    vis = img.copy()

    for x1, y1, x2, y2 in crops:
        # Draw main rectangle
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        w = x2 - x1
        h = y2 - y1

        # Corner marker length proportional to size
        cl = int(min(w, h) * corner_length_ratio)

        # Top-left corner
        cv2.line(vis, (x1, y1), (x1 + cl, y1), color, corner_thickness)
        cv2.line(vis, (x1, y1), (x1, y1 + cl), color, corner_thickness)

        # Top-right corner
        cv2.line(vis, (x2, y1), (x2 - cl, y1), color, corner_thickness)
        cv2.line(vis, (x2, y1), (x2, y1 + cl), color, corner_thickness)

        # Bottom-left corner
        cv2.line(vis, (x1, y2), (x1 + cl, y2), color, corner_thickness)
        cv2.line(vis, (x1, y2), (x1, y2 - cl), color, corner_thickness)

        # Bottom-right corner
        cv2.line(vis, (x2, y2), (x2 - cl, y2), color, corner_thickness)
        cv2.line(vis, (x2, y2), (x2, y2 - cl), color, corner_thickness)

    return vis
