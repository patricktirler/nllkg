import numpy as np
from typing import List
from dataclasses import dataclass
from scipy.optimize import minimize


@dataclass
class ShapeTemplate:
    """
    Defines a shape template with keypoints and their initial coordinates.
    
    Attributes:
        keypoint_coords: (M, 2) array of template keypoint coordinates.
        keypoint_label_names: (M,) array of keypoint labels corresponding to the template keypoints.
    """
    keypoint_coords: np.ndarray
    keypoint_label_names: np.ndarray


def fit_shapes_with_keypoints(
    keypoint_scores: np.ndarray,
    keypoint_coords: np.ndarray,
    keypoint_label_names: np.ndarray,
    template_keypoints: List[ShapeTemplate],
    dof: str = "trs",
    keypoint_score_threshold: float = 0.3,
    sigma: float = 10.0,
) -> List[ShapeTemplate]:
    """
    Refine shape templates to match observed keypoints.
    
    Args:
        keypoint_scores: (N,) array of keypoint confidence scores (0-1).
        keypoint_coords: (N, 2) array of observed keypoint coordinates.
        keypoint_label_names: (N,) array of observed keypoint labels.
        template_keypoints: List of ShapeTemplate objects to refine.
        dof: Degrees of freedom as string:
            - "t": translation only
            - "tr": translation + rotation
            - "trs": translation + rotation + uniform scale (default)
            - "trsxsy": translation + rotation + scale x/y
            - "perspective": 2x3 affine perspective transform
        keypoint_score_threshold: Minimum keypoint confidence to use.
        sigma: Gaussian bandwidth for likelihood.
    
    Returns:
        List of refined ShapeTemplate objects.
    """
    # Filter observations by score threshold
    keep = keypoint_scores >= keypoint_score_threshold
    keypoint_coords = keypoint_coords[keep]
    keypoint_scores = keypoint_scores[keep]
    keypoint_label_names = keypoint_label_names[keep]
        
    # Refine each template
    refined_templates = []
    for template in template_keypoints:
        refined = _refine_single_template(
            template, keypoint_coords, keypoint_scores, keypoint_label_names, dof, sigma
        )
        refined_templates.append(refined)
    
    return refined_templates


def _refine_single_template(
    template: ShapeTemplate,
    obs_coords: np.ndarray,
    obs_scores: np.ndarray,
    obs_label_names: np.ndarray,
    dof: str,
    sigma: float,
) -> ShapeTemplate:
    """Refine a single template against observations."""
    dof_list = _parse_dof(dof)
    initial_params = _dof_defaults(dof_list)
    
    template_coords = template.keypoint_coords
    template_label_names = template.keypoint_label_names
    
    def objective(params):
        T = _params_to_matrix(params, dof)
        
        # Transform template points
        template_h = np.hstack([template_coords, np.ones((template_coords.shape[0], 1))])
        transformed = template_h @ T.T  # (M, 2)
        
        # Create label matching mask: (M, N)
        label_mask = template_label_names[:, np.newaxis] == obs_label_names[np.newaxis, :]  # (M, N)
        
        # Compute distances: (M, N)
        diff = transformed[:, np.newaxis, :] - obs_coords[np.newaxis, :, :]  # (M, N, 2)
        dist_sq = np.sum(diff**2, axis=-1)  # (M, N)
        
        # Compute likelihoods and zero out non-matching labels
        likelihoods = obs_scores * np.exp(-dist_sq / (2 * sigma**2))  # (M, N)
        likelihoods = likelihoods * label_mask  # (M, N)
        
        return -np.sum(likelihoods)
    
    opt_result = minimize(objective, initial_params, method="L-BFGS-B",
                         options={"maxiter": 100})
    
    # Transform template with optimized parameters
    T = _params_to_matrix(opt_result.x, dof)
    template_h = np.hstack([template_coords, np.ones((template_coords.shape[0], 1))])
    refined_coords = template_h @ T.T
    
    return ShapeTemplate(
        keypoint_coords=refined_coords,
        keypoint_label_names=template_label_names
    )


def _parse_dof(dof: str) -> List[str]:
    """Parse DOF string into list of parameters."""
    mapping = {
        "t": ["tx", "ty"],
        "tr": ["tx", "ty", "theta"],
        "trs": ["tx", "ty", "theta", "s"],
        "trsxsy": ["tx", "ty", "theta", "sx", "sy"],
        "perspective": ["a", "b", "c", "d", "tx", "ty"],
    }
    if dof not in mapping:
        raise ValueError(f"Unknown DOF: {dof}. Options: {list(mapping.keys())}")
    return mapping[dof]


def _dof_defaults(dof_list: List[str]) -> np.ndarray:
    """Get initial parameter values for DOF."""
    defaults = {
        "tx": 0.0, "ty": 0.0, "theta": 0.0,
        "s": 1.0, "sx": 1.0, "sy": 1.0,
        "a": 1.0, "b": 0.0, "c": 0.0, "d": 1.0,
    }
    return np.array([defaults[d] for d in dof_list])


def _params_to_matrix(params: np.ndarray, dof: str) -> np.ndarray:
    """Convert parameters to 2x3 transformation matrix."""
    dof_list = _parse_dof(dof)
    param_dict = dict(zip(dof_list, params))
    
    if dof == "perspective":
        # Direct 2x3 affine: [a, b, c, d, tx, ty]
        a = param_dict.get("a", 1.0)
        b = param_dict.get("b", 0.0)
        c = param_dict.get("c", 0.0)
        d = param_dict.get("d", 1.0)
        tx = param_dict.get("tx", 0.0)
        ty = param_dict.get("ty", 0.0)
        return np.array([[a, b, tx], [c, d, ty]])
    
    # Compose from basic transforms
    tx = param_dict.get("tx", 0.0)
    ty = param_dict.get("ty", 0.0)
    theta = param_dict.get("theta", 0.0)
    
    # Determine scale
    if "s" in param_dict:
        sx = sy = param_dict["s"]
    else:
        sx = param_dict.get("sx", 1.0)
        sy = param_dict.get("sy", 1.0)
    
    # Scale + Rotate + Translate
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [sx * c, -sy * s, tx],
        [sx * s,  sy * c, ty],
    ])