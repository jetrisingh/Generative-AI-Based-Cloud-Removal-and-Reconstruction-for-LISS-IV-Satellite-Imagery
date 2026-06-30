"""
metrics.py
----------
Small, dependency-light metrics used to populate the UI's metrics row.
None of these require your model internals -- they're computed purely
from the images returned by model_pipeline.run_pipeline().
"""

import numpy as np
from PIL import Image, ImageFilter


def cloud_coverage_pct(cloud_mask: np.ndarray) -> float:
    """Percentage of pixels flagged as cloud by stage 1."""
    if cloud_mask is None:
        return None
    return round(float(cloud_mask.mean()) * 100, 1)


def _laplacian_variance(image: np.ndarray) -> float:
    """
    A simple, OpenCV-free sharpness proxy: variance of a Laplacian-like
    edge filter. Higher = sharper / more high-frequency detail.
    """
    gray = Image.fromarray(image).convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    arr = np.asarray(edges).astype(np.float32)
    return float(arr.var())


def sharpness_gain_pct(before: np.ndarray, after: np.ndarray) -> float:
    """Relative change in sharpness between two stages, as a percentage."""
    v_before = _laplacian_variance(before)
    v_after = _laplacian_variance(after)
    if v_before <= 1e-6:
        return None
    gain = ((v_after - v_before) / v_before) * 100
    return round(gain, 1)


def psnr_db(reference: np.ndarray, comparison: np.ndarray) -> float:
    """
    Standard PSNR between two same-sized images. In production, `reference`
    would ideally be a known-clean ground truth tile (e.g. for a held-out
    validation set); here we compare reconstructed vs. cloud-removed as an
    illustrative proxy since no ground truth is available at inference time.
    """
    if reference.shape != comparison.shape:
        return None
    mse = np.mean((reference.astype(np.float32) - comparison.astype(np.float32)) ** 2)
    if mse == 0:
        return 99.0
    return round(float(20 * np.log10(255.0 / np.sqrt(mse))), 1)
