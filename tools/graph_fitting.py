import numpy as np
from typing import List
from scipy.optimize import minimize
from abc import ABC, abstractmethod

class ShapeTemplate(ABC):
    """
    Base class for shapes. 
    Subclasses must define how to generate coordinates from a parameter vector.
    """
    def __init__(self, label_names: np.ndarray):
        self.keypoint_label_names = label_names

    @abstractmethod
    def get_params(self) -> np.ndarray:
        """Serialize current geometric state to a numpy array."""
        pass

    @abstractmethod
    def from_params(self, params: np.ndarray) -> 'ShapeTemplate':
        """Create a new instance of this shape from a parameter array."""
        pass

    @abstractmethod
    def get_coords_from_params(self, params: np.ndarray) -> np.ndarray:
        """Calculate keypoint coordinates for a given set of parameters."""
        pass

    @property
    def keypoint_coords(self) -> np.ndarray:
        """Current coordinates based on internal state."""
        return self.get_coords_from_params(self.get_params())



def fit_shapes(
    templates: List[ShapeTemplate],
    keypoint_coords: np.ndarray,
    keypoint_scores: np.ndarray,
    keypoint_label_names: np.ndarray,
    sigma: float = 10.0,
    keypoint_score_threshold: float = 0.3
) -> List[ShapeTemplate]:
    """
    Fits provided shape objects to observed keypoints.
    """
    # Filter noise
    mask = keypoint_scores >= keypoint_score_threshold
    obs_c, obs_s, obs_l = keypoint_coords[mask], keypoint_scores[mask], keypoint_label_names[mask]

    refined_results = []

    for template in templates:
        initial_params = template.get_params()
        # Pre-compute label matches for this template
        label_mask = template.keypoint_label_names[:, np.newaxis] == obs_l[np.newaxis, :]

        def objective(params):
            coords = template.get_coords_from_params(params)
            
            # Distance squared (M, N)
            diff = coords[:, np.newaxis, :] - obs_c[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=-1)
            
            # Soft-matching Gaussian likelihood
            likelihoods = obs_s * np.exp(-dist_sq / (2 * sigma**2))
            return -np.sum(likelihoods * label_mask)

        
        res = minimize(objective, initial_params, method="L-BFGS-B")
        refined_results.append(template.from_params(res.x))

    return refined_results