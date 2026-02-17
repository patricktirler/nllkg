import numpy as np
from typing import Any
from scipy.optimize import minimize
from abc import ABC, abstractmethod
import copy


class ShapeTemplate(ABC):
    """
    Abstract base class representing a parametric geometric shape template.
    """

    def __init__(self, params: np.ndarray, label_names: np.ndarray):
        """
        Initialize a shape template.

        Parameters
        ----------
        params : np.ndarray of shape (P,)
            Parameter vector describing a specific geometric configuration.
        label_names : np.ndarray of shape (M,)
            Array of keypoint label identifiers associated with this template.
            These must match the label space used by detected keypoints.
        """
        self.params = params
        self.keypoint_label_names = label_names


    @abstractmethod
    def get_coords_from_params(self, params: np.ndarray) -> np.ndarray:
        """
        Compute keypoint coordinates from a parameter vector.

        Parameters
        ----------
        params : np.ndarray of shape (P,)
            Parameter vector describing a geometric configuration.

        Returns
        -------
        np.ndarray of shape (M, 2)
            2D coordinates of all template keypoints.
            M is the number of template keypoints.
        """
        pass


    def fit(
        self,
        keypoint_coords: np.ndarray,
        keypoint_scores: np.ndarray,
        keypoint_label_names: np.ndarray,
        sigma: float = 10.0,
        method: str = "L-BFGS-B",
        **minimize_kwargs: Any,
    ) -> "ShapeTemplate":
        """
        Fit this template to observed keypoints using soft Gaussian matching.

        Parameters
        ----------
        keypoint_coords : np.ndarray of shape (N, 2)
            Observed 2D keypoint coordinates.

        keypoint_scores : np.ndarray of shape (N,)
            Confidence scores for each observed keypoint.
            Used as weights in the likelihood computation.

        keypoint_label_names : np.ndarray of shape (N,)
            Label identifiers for observed keypoints.
            Must be comparable with `self.keypoint_label_names`.

        sigma : float, default=10.0
            Standard deviation of the Gaussian kernel controlling spatial tolerance.

        method : str, default="L-BFGS-B"
            Optimization algorithm passed to `scipy.optimize.minimize`.

        **minimize_kwargs : dict
            Additional keyword arguments forwarded directly to `scipy.optimize.minimize`.

        Returns
        -------
        ShapeTemplate
            A new shape instance with optimized parameters.
        """

        initial_params = self.params

        # Precompute label matching mask (M template × N observed)
        label_mask = (
            self.keypoint_label_names[:, np.newaxis]
            == keypoint_label_names[np.newaxis, :]
        )

        def objective(params: np.ndarray) -> float:
            coords = self.get_coords_from_params(params)

            # Pairwise squared distances (M, N)
            diff = coords[:, np.newaxis, :] - keypoint_coords[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=-1)

            likelihoods = keypoint_scores * np.exp(-dist_sq / (2 * sigma**2))

            return -np.sum(likelihoods * label_mask)

        result = minimize(
            objective,
            initial_params,
            method=method,
            **minimize_kwargs,
        )

        new_instance = copy.deepcopy(self)
        new_instance.params = result.x
        new_instance.opt_result = result
        return new_instance