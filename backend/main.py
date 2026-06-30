"""
main.py
-------
FastAPI backend for the ISRO-CRR demo.

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Endpoints:
    GET  /api/health   -> liveness check (drives the frontend's STATUS pill)
    POST /api/process  -> multipart/form-data with a "file" field.
                          Returns the cloud-removed + reconstructed display
                          images as base64 PNGs, plus computed metrics.

Band handling
-------------
The model expects 3 bands in the strict order Green, Red, NIR (LISS-IV
Bands 2/3/4). Two upload paths are supported:

  1. Multi-band TIFF (real LISS-IV tiles) -- read via tifffile, assumed
     to already be stored as exactly 3 bands in Green, Red, NIR order.
     If your TIFFs use a different band order, fix it in
     `_extract_bands_from_tiff()` below.

  2. Ordinary PNG/JPG (for demos without real multiband data) -- there
     is no real NIR channel in a normal photo, so one is approximated
     from the visible bands. This is a DEMO-ONLY fallback, clearly
     marked below; don't use it to draw conclusions about real model
     accuracy.
"""

import io
import time

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from model_pipeline import run_pipeline
import metrics as metrics_mod

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

app = FastAPI(title="ISRO-CRR API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your frontend's actual domain before going public
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
MAX_BYTES = 25 * 1024 * 1024  # 25 MB, matches frontend copy
MAX_DISPLAY_DIM = 1024  # cap full-res tiles for demo responsiveness


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/process")
async def process_image(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()
    if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Unsupported file type. Use TIFF, PNG, or JPG.")

    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds 25MB limit.")

    try:
        bands_grn = _extract_bands(raw, filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    # Cap resolution for demo responsiveness; raise/remove for full-res runs.
    bands_grn = _cap_resolution(bands_grn, MAX_DISPLAY_DIM)

    start = time.time()
    try:
        result = run_pipeline(bands_grn)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")
    elapsed = time.time() - start

    metrics = {
        "cloud_coverage_pct": metrics_mod.cloud_coverage_pct(result["cloud_mask"]),
        "sharpness_gain_pct": metrics_mod.sharpness_gain_pct(
            result["original_rgb"], result["reconstructed_rgb"]
        ),
        "psnr_db": metrics_mod.psnr_db(result["cloud_removed_rgb"], result["reconstructed_rgb"]),
        "inference_time_s": round(elapsed, 2),
    }

    return {
        "original_image": _to_data_url(result["original_rgb"]),
        "cloud_removed_image": _to_data_url(result["cloud_removed_rgb"]),
        "reconstructed_image": _to_data_url(result["reconstructed_rgb"]),
        "metrics": metrics,
    }


# =====================================================================
# BAND EXTRACTION
# =====================================================================
def _extract_bands(raw: bytes, filename: str) -> np.ndarray:
    """Returns (H, W, 3) float32 in [0,1], channel order Green/Red/NIR."""
    if filename.endswith(".tif") or filename.endswith(".tiff"):
        return _extract_bands_from_tiff(raw)
    return _extract_bands_from_photo(raw)


def _extract_bands_from_tiff(raw: bytes) -> np.ndarray:
    """
    Real LISS-IV path. Assumes the TIFF already stores exactly 3 bands
    in Green, Red, NIR order (matching the model's expected channel
    order). Adjust the indexing below if your data is laid out
    differently (e.g. extra bands, or a different channel order).

    Returns RAW float32 values (NOT pre-scaled to [0,1]) -- the
    pipeline's percentile_normalize() applies the same per-band 2nd/98th
    percentile stretch used in the team's training/inference scripts, so
    raw sensor values need to reach it unmodified for results to match.
    """
    if not TIFFFILE_AVAILABLE:
        raise RuntimeError("tifffile is not installed -- add it to requirements.txt to read multiband TIFFs.")

    arr = tifffile.imread(io.BytesIO(raw))

    # Normalize to (H, W, C)
    if arr.ndim == 2:
        raise RuntimeError("TIFF has only 1 band; expected 3 (Green, Red, NIR).")
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))  # (C,H,W) -> (H,W,C)

    if arr.shape[-1] < 3:
        raise RuntimeError(f"Expected >= 3 bands, got shape {arr.shape}.")

    return arr[..., :3].astype(np.float32)  # Green, Red, NIR -- first 3 bands, raw values


def _extract_bands_from_photo(raw: bytes) -> np.ndarray:
    """
    DEMO-ONLY fallback for ordinary PNG/JPG uploads (no real NIR band
    available). Approximates:
        Green <- photo's G channel
        Red   <- photo's R channel
        NIR   <- synthesized from R/G (NOT real NIR -- for UI demos only)

    Returns values scaled to roughly [0,255] (not [0,1]) so they pass
    through the pipeline's percentile_normalize() the same way real
    sensor values would.

    Swap your own NIR-bearing source (or skip this path entirely and
    require multiband TIFF uploads) for any result you intend to show
    as a real accuracy demo.
    """
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    rgb = np.asarray(pil_img).astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    green = g
    red = r
    nir_proxy = np.clip(255.0 - b * 0.5 + (r - g) * 0.3, 0.0, 255.0)  # crude vegetation-ish proxy

    bands = np.stack([green, red, nir_proxy], axis=-1)
    return bands.astype(np.float32)


def _cap_resolution(bands: np.ndarray, max_dim: int) -> np.ndarray:
    """Resizes down if needed, preserving raw value range (works for
    both [0,1]-ish photo proxies and raw 16-bit sensor ranges)."""
    h, w = bands.shape[:2]
    if max(h, w) <= max_dim:
        return bands
    scale = max_dim / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    channels = [
        np.asarray(
            Image.fromarray(bands[..., c].astype(np.float32), mode="F").resize(
                (new_w, new_h), Image.BILINEAR
            ),
            dtype=np.float32,
        )
        for c in range(bands.shape[-1])
    ]
    return np.stack(channels, axis=-1)


def _to_data_url(arr: np.ndarray) -> str:
    img = Image.fromarray(arr.astype("uint8"))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    import base64
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"
