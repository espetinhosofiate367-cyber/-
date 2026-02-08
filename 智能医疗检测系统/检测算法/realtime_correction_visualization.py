"""
Realtime Correction-Visualization Integrated Algorithm Module
=============================================================
This module implements the core steps described in the patent draft
"Real-time Correction-Visualization Integrated Algorithm and System for
Multi-dimensional Biosensors".  The focus is to provide a light-weight,
self-contained implementation that can be called by GUI programs such as
``modern_detection_gui_optimized.py``.

Pipeline Overview
-----------------
For each incoming sensor frame (2-D pressure / stress matrix):
1. Data preprocessing & cleaning
2. GMM-based 3-D probabilistic heat-map reconstruction
3. Lightweight CNN / ConvLSTM like correction (approximated here)
4. Self-supervised enhancement (denoising + sharpening)
5. Dynamic confidence calibration
6. Morphological + statistical post-processing

The implementation below chooses pragmatic, dependency-friendly
strategies so that it works out-of-box with the existing project
requirements.  Heavyweight deep-learning parts are approximated with
numerical filters to keep the example easily runnable ‑ users can swap
in their own Keras / PyTorch models as needed.

Author: Trae-AI assistant
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.mixture import GaussianMixture
import cv2  # type: ignore
from typing import Tuple, Dict, Any

# ----------------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------------

def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise an array to 0-1 range (safe for constant arrays)."""
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax - vmin < 1e-8:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)


def _resize(arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize 2-D array using cubic interpolation (OpenCV)."""
    return cv2.resize(arr, size[::-1], interpolation=cv2.INTER_CUBIC)

# ----------------------------------------------------------------------------
# Core algorithm class
# ----------------------------------------------------------------------------


class RealtimeCorrectionVisualizationSystem:
    """High-level façade for the realtime correction & visualization pipeline."""

    def __init__(self,
                 grid_size: Tuple[int, int] = (8, 8),
                 n_gmm_components: int = 3,
                 smooth_sigma: float = 1.5,
                 conf_percentiles: Tuple[float, float] = (5, 95)) -> None:
        self.grid_size = grid_size
        self.n_gmm_components = n_gmm_components
        self.smooth_sigma = smooth_sigma
        self.low_pct, self.high_pct = conf_percentiles

        # Store previous heat-map for simple temporal smoothing (ConvLSTM stub)
        self._prev_heatmap: np.ndarray | None = None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def process_frame(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Process a single sensor matrix and return dictionary with results.

        Returns
        -------
        dict with keys:
            'corrected'   : 2-D np.ndarray of corrected intensity
            'raw_heatmap' : 2-D np.ndarray of GMM heat-map (0-1)
            'confidence'  : float, dynamic calibrated confidence metric
        """
        if matrix.shape != self.grid_size:
            matrix = _resize(matrix, self.grid_size)

        clean = self._preprocess(matrix)
        raw_heat = self._gmm_heatmap(clean)
        corrected = self._cnn_correction(raw_heat)
        enhanced = self._self_supervised_enhance(corrected)
        final_map, conf = self._dynamic_calibrate(enhanced)
        final_map = self._post_process(final_map)

        return {
            'corrected': final_map,
            'raw_heatmap': raw_heat,
            'confidence': conf,
        }

    # ------------------------------------------------------------------
    # Individual pipeline steps
    # ------------------------------------------------------------------

    def _preprocess(self, matrix: np.ndarray) -> np.ndarray:
        """Simple cleaning: clip outliers & normalise."""
        # Median filtering for salt-pepper noise.
        filtered = ndimage.median_filter(matrix, size=3)
        # Clip to percentile range
        low, high = np.percentile(filtered, [1, 99])
        clipped = np.clip(filtered, low, high)
        return _normalize(clipped)

    def _gmm_heatmap(self, matrix: np.ndarray) -> np.ndarray:
        """Fit a GMM and reconstruct probabilistic heat-map."""
        h, w = matrix.shape
        yy, xx = np.mgrid[0:h, 0:w]
        coords = np.column_stack([xx.ravel(), yy.ravel()])
        values = matrix.ravel().reshape(-1, 1)

        # Weight the coordinates by intensity to bias clusters
        weights = values.flatten() + 1e-6  # avoid zeros
        gmm = GaussianMixture(n_components=self.n_gmm_components, covariance_type='diag',
                              max_iter=100, random_state=0)
        # Fit GMM (handle old scikit-learn versions without sample_weight)
        try:
            gmm.fit(coords, sample_weight=weights)
        except TypeError:
            # sample_weight unsupported → fall back to unweighted fit
            gmm.fit(coords)

        # Compute density over grid
        log_prob = gmm.score_samples(coords)
        heatmap = np.exp(log_prob).reshape(h, w)
        heatmap = _normalize(heatmap)
        # Optional smoothing
        heatmap = ndimage.gaussian_filter(heatmap, sigma=self.smooth_sigma)
        return heatmap

    def _cnn_correction(self, heatmap: np.ndarray) -> np.ndarray:
        """Lightweight CNN / ConvLSTM analogue.

        To avoid heavyweight deep-learning deps, we approximate with a
        separable Sobel edge-aware enhancement + temporal smoothing.
        """
        # Edge emphasis (like early CNN conv layer)
        sobel_x = ndimage.sobel(heatmap, axis=1)
        sobel_y = ndimage.sobel(heatmap, axis=0)
        edges = np.hypot(sobel_x, sobel_y)

        # Combine with original heatmap
        combined = 0.8 * heatmap + 0.2 * _normalize(edges)

        # Temporal smoothing (ConvLSTM stub)
        if self._prev_heatmap is not None:
            combined = 0.6 * combined + 0.4 * self._prev_heatmap
        self._prev_heatmap = combined.copy()
        return combined

    def _self_supervised_enhance(self, heatmap: np.ndarray) -> np.ndarray:
        """Denoising auto-augmentation (self-supervised stub)."""
        # Sharpen via unsharp mask
        blurred = ndimage.gaussian_filter(heatmap, sigma=1)
        sharpened = np.clip(heatmap + (heatmap - blurred), 0, 1)
        return sharpened

    def _dynamic_calibrate(self, heatmap: np.ndarray) -> Tuple[np.ndarray, float]:
        """Scale heatmap so that selected percentile spans 0-1.  Return conf."""
        low_val, high_val = np.percentile(heatmap, [self.low_pct, self.high_pct])
        calibrated = np.clip((heatmap - low_val) / (high_val - low_val + 1e-6), 0, 1)
        confidence = float((high_val - low_val))  # spread as proxy for certainty
        return calibrated, confidence

    def _post_process(self, heatmap: np.ndarray) -> np.ndarray:
        """Morphological closing + small object removal."""
        # Convert to 8-bit for OpenCV morphology
        img8 = (heatmap * 255).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(img8, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Remove tiny islands using connected-component threshold
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        min_area = 4  # pixels
        mask = np.zeros_like(closed, dtype=bool)
        for i in range(1, num_labels):  # skip background
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                mask |= labels == i
        processed = np.where(mask, closed, 0).astype(np.float32) / 255.0
        return processed

# -----------------------------------------------------------------------------
# Quick self-test (only run when executed directly)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate synthetic test frame (two Gaussian blobs)
    y, x = np.mgrid[0:8, 0:8]
    blob1 = np.exp(-((x - 2) ** 2 + (y - 3) ** 2) / 4)
    blob2 = 0.8 * np.exp(-((x - 6) ** 2 + (y - 5) ** 2) / 6)
    frame = 100 * (blob1 + blob2)  # scale to 0-100 arbitrary units

    system = RealtimeCorrectionVisualizationSystem()
    result = system.process_frame(frame)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Raw Heatmap")
    plt.imshow(result['raw_heatmap'], cmap='turbo')
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.title("Corrected")
    plt.imshow(result['corrected'], cmap='turbo')
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.title(f"Confidence: {result['confidence']:.3f}")
    plt.imshow(result['corrected'] > 0.6, cmap='gray')
    plt.tight_layout()
    plt.show()