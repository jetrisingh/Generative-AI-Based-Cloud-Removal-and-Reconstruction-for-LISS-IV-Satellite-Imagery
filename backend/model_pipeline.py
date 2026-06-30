"""
model_pipeline.py
------------------
Integrates the real CloudRemovalUNet (Model.py, by Jetri's teammate) into
the app, matching the exact contract used in their inference.py /
ReconsUnet.py training script.

MODEL CONTRACT (confirmed from their code)
-------------------------------------------
Input  : torch.FloatTensor, shape (B, 4, H, W), values in [0, 1]
         channel 0 -> Green (LISS-IV Band 2)
         channel 1 -> Red   (LISS-IV Band 3)
         channel 2 -> NIR   (LISS-IV Band 4)
         channel 3 -> Cloud mask (binary: 0.0 clear, 1.0 cloud)
         H = W = 256 (fixed, training-matched)
Device : CUDA if available, else CPU
Output : (B, 3, H, W) float32 in [0,1] -- model has a Sigmoid head, so
         no extra clamping/scaling is needed on the way out.

Checkpoint format (from ReconsUnet.py's torch.save call):
    state = {"epoch": ..., "net": net.state_dict(), "opt": ...}
    torch.save(state, "unet_best.pth")
So loading requires ckpt["net"], not the raw checkpoint dict.

Normalization (from DatastePrep.py / inference.py's read_tif):
    Per-band 2nd/98th percentile stretch, computed per image:
        out = clip((band - p2) / (p98 - p2 + 1e-6), 0, 1)
    This is NOT a simple /255 division -- using the wrong normalization
    here will visibly degrade output quality even with correct weights,
    so it's reproduced exactly in `percentile_normalize()` below.
"""

import os
import numpy as np
from PIL import Image

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from Model import CloudRemovalUNet

# =====================================================================
# CONFIG
# =====================================================================
MODEL_INPUT_SIZE = 256  # fixed, training-matched -- do not change

MODEL_CHECKPOINT_PATH = os.environ.get(
    "MODEL_CHECKPOINT_PATH",
    os.path.join(os.path.dirname(__file__), "weights", "unet_best.pth"),
)

# IMPORTANT: must match what was used when this checkpoint was trained.
# Model.py's class default is 64; ReconsUnet.py's training script CLI
# default is 32. If your teammate didn't pass --base_ch when training,
# it's 32. Get this confirmed -- a mismatch here causes a clear
# shape-mismatch error on load (see _load_model() below), not silent
# wrong output, so it's easy to detect and fix.
MODEL_BASE_CH = int(os.environ.get("MODEL_BASE_CH", "32"))

DEVICE = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

_model = None
_model_load_attempted = False


def _load_model():
    """Loads CloudRemovalUNet + checkpoint once. Returns None if
    unavailable, in which case run_model() falls back to a DEMO
    implementation so the rest of the app keeps working."""
    global _model, _model_load_attempted
    if _model_load_attempted:
        return _model
    _model_load_attempted = True

    if not TORCH_AVAILABLE:
        print("[model_pipeline] torch not installed -- running DEMO fallback.")
        return None
    if not os.path.exists(MODEL_CHECKPOINT_PATH):
        print(f"[model_pipeline] No checkpoint at {MODEL_CHECKPOINT_PATH} -- running DEMO fallback.")
        return None

    try:
        net = CloudRemovalUNet(in_ch=4, out_ch=3, base_ch=MODEL_BASE_CH).to(DEVICE)
        ckpt = torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE)

        # Their checkpoints are saved as {"epoch":..., "net": state_dict, "opt":...}.
        # Support both that format and a raw state_dict, just in case.
        state_dict = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
        net.load_state_dict(state_dict)

        net.eval()
        _model = net
        print(f"[model_pipeline] Loaded CloudRemovalUNet (base_ch={MODEL_BASE_CH}) "
              f"from {MODEL_CHECKPOINT_PATH} on {DEVICE}.")
    except RuntimeError as e:
        # Most common cause: MODEL_BASE_CH doesn't match the checkpoint's
        # trained width -- the error message will say so explicitly
        # (e.g. "size mismatch for enc1...").
        print(f"[model_pipeline] Failed to load model ({e})")
        print("[model_pipeline] If this is a size-mismatch error, check MODEL_BASE_CH "
              "matches --base_ch used during training (default 32 in ReconsUnet.py).")
        print("[model_pipeline] Running DEMO fallback.")
        _model = None
    except Exception as e:
        print(f"[model_pipeline] Failed to load model ({e}) -- running DEMO fallback.")
        _model = None

    return _model


# =====================================================================
# NORMALIZATION -- must match DatastePrep.py / inference.py exactly
# =====================================================================
def percentile_normalize(bands_raw: np.ndarray) -> np.ndarray:
    """
    bands_raw: (H, W, 3) float32, RAW sensor values (not yet in [0,1]).
    Returns (H, W, 3) float32 in [0,1], using the same per-band 2nd/98th
    percentile stretch as their read_tif().

    Note: if your upload path already hands you values pre-scaled to
    [0,1] (e.g. an 8-bit PNG/JPG divided by 255), this still works --
    it just re-stretches contrast the same way their real data pipeline
    does, which keeps inference consistent with how the model was
    trained.
    """
    out = np.zeros_like(bands_raw, dtype=np.float32)
    for b in range(bands_raw.shape[2]):
        band = bands_raw[:, :, b]
        positive = band[band > 0]
        if positive.size > 0:
            p2, p98 = np.percentile(positive, (2, 98))
            p2, p98 = np.float32(p2), np.float32(p98)
        else:
            p2, p98 = np.float32(0), np.float32(1)
        out[:, :, b] = np.clip((band - p2) / (p98 - p2 + np.float32(1e-6)), 0, 1)
    return out


# =====================================================================
# CLOUD MASK (channel 3 of the model input)
# =====================================================================
def detect_cloud_mask(bands_norm: np.ndarray, threshold: float = 0.75) -> np.ndarray:
    """
    bands_norm: (H, W, 3) float32 in [0,1], already percentile-normalized.
    Returns a BINARY (H, W) float32 mask: 1.0 = cloud, 0.0 = clear.

    Reproduces inference.py's auto_cloud_mask(): bright across all bands
    = likely cloud, then morphologically dilated. Uses PIL instead of
    cv2's dilate to avoid adding an extra dependency; behavior is
    equivalent for this purpose.

    If your teammate hands you real QGIS-made cloud masks for specific
    demo tiles, prefer loading those directly instead of this heuristic
    -- swap this function's call site in run_pipeline() accordingly.
    """
    bright = np.all(bands_norm > threshold, axis=2).astype(np.float32)
    mask_img = Image.fromarray((bright * 255).astype(np.uint8))
    # Dilate via max-filter as a dependency-free stand-in for cv2.dilate
    from PIL import ImageFilter
    dilated = mask_img.filter(ImageFilter.MaxFilter(size=11))
    return (np.asarray(dilated).astype(np.float32) / 255.0 > 0.5).astype(np.float32)


def visualize_masked_input(bands_norm: np.ndarray, cloud_mask: np.ndarray) -> np.ndarray:
    """Zeroes out cloud pixels for the UI's middle ('cloud removed' /
    occlusion) panel. Visualization only -- not a model call."""
    return bands_norm * (1 - cloud_mask[..., None])


# =====================================================================
# PRE/POST-PROCESSING around the model call
# =====================================================================
def _resize_array(arr: np.ndarray, size: int, is_mask: bool) -> np.ndarray:
    """Resize (H,W) or (H,W,C) float32 [0,1] array to (size,size),
    keeping float precision via mode='F' instead of round-tripping
    through uint8."""
    resample = Image.NEAREST if is_mask else Image.BILINEAR

    if arr.ndim == 2:
        img = Image.fromarray(arr.astype(np.float32), mode="F")
        return np.asarray(img.resize((size, size), resample=resample), dtype=np.float32)

    channels = [
        np.asarray(
            Image.fromarray(arr[..., c].astype(np.float32), mode="F").resize(
                (size, size), resample=resample
            ),
            dtype=np.float32,
        )
        for c in range(arr.shape[-1])
    ]
    return np.stack(channels, axis=-1)


def preprocess(bands_norm: np.ndarray, cloud_mask: np.ndarray):
    """
    Builds the (1, 4, 256, 256) tensor the model expects, matching
    infer_scene()'s torch.cat([img_t, mask_t], dim=1) channel order:
    image bands first (Green, Red, NIR), mask last.
    """
    original_size = (bands_norm.shape[1], bands_norm.shape[0])  # (W, H)

    bands_resized = _resize_array(bands_norm, MODEL_INPUT_SIZE, is_mask=False)
    mask_resized = _resize_array(cloud_mask, MODEL_INPUT_SIZE, is_mask=True)
    mask_resized = (mask_resized > 0.5).astype(np.float32)

    stacked = np.concatenate([bands_resized, mask_resized[..., None]], axis=-1)  # (256,256,4)
    stacked = np.clip(stacked, 0.0, 1.0).astype(np.float32)
    chw = np.transpose(stacked, (2, 0, 1))  # (4,256,256)
    batched = chw[None, ...]  # (1,4,256,256)

    if TORCH_AVAILABLE:
        return torch.from_numpy(batched).float(), original_size
    return batched, original_size


def run_model(input_tensor):
    """
    Runs CloudRemovalUNet if a checkpoint is loaded, else a DEMO
    fallback. input_tensor: (1,4,256,256) torch.Tensor (or numpy array
    if torch isn't installed -- demo-only path).
    Returns (1,3,256,256) float32 numpy array in [0,1].
    """
    model = _load_model()

    if model is not None:
        with torch.no_grad():
            output = model(input_tensor.to(DEVICE))
        # Model already ends in Sigmoid, so output is in [0,1] -- clamp
        # only as a safety net against numerical edge cases.
        return output.detach().cpu().clamp(0, 1).numpy().astype(np.float32)

    # --- DEMO fallback (checkpoint not found yet) ---
    from PIL import ImageFilter, ImageEnhance
    arr = input_tensor.numpy() if TORCH_AVAILABLE else input_tensor
    bands = arr[0, :3]
    mask = arr[0, 3]

    hwc = np.transpose(bands, (1, 2, 0))
    pil_img = Image.fromarray((np.clip(hwc, 0, 1) * 255).astype(np.uint8))
    blurred_arr = np.asarray(pil_img.filter(ImageFilter.GaussianBlur(radius=10))).astype(np.float32) / 255.0
    m = mask[..., None]
    filled = hwc * (1 - m) + blurred_arr * m

    filled_pil = Image.fromarray((np.clip(filled, 0, 1) * 255).astype(np.uint8))
    sharpened = filled_pil.filter(ImageFilter.UnsharpMask(radius=2, percent=140, threshold=2))
    sharpened = ImageEnhance.Contrast(sharpened).enhance(1.05)
    out_hwc = np.asarray(sharpened).astype(np.float32) / 255.0

    return np.transpose(out_hwc, (2, 0, 1))[None, ...].astype(np.float32)


def postprocess_output(output_array: np.ndarray, original_size) -> np.ndarray:
    """
    output_array: (1,3,256,256) float32 [0,1], Green/Red/NIR order.
    original_size: (W,H) to resize back up to.
    Returns (H,W,3) float32 [0,1] bands, order Green/Red/NIR.
    """
    bands_256 = np.transpose(output_array[0], (1, 2, 0))
    w, h = original_size
    channels = [
        np.asarray(
            Image.fromarray((bands_256[..., c] * 255).astype(np.uint8)).resize(
                (w, h), resample=Image.BILINEAR
            ),
            dtype=np.float32,
        )
        / 255.0
        for c in range(3)
    ]
    return np.stack(channels, axis=-1)


# =====================================================================
# DISPLAY -- map Green/Red/NIR bands to an RGB image browsers can show
# =====================================================================
def bands_to_display_rgb(bands_grn: np.ndarray) -> np.ndarray:
    """
    LISS-IV has no blue band, so true color isn't possible. Uses the
    standard False Color Composite convention:
        display R <- NIR, display G <- Red, display B <- Green
    """
    g, r, nir = bands_grn[..., 0], bands_grn[..., 1], bands_grn[..., 2]
    rgb = np.stack([nir, r, g], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


# =====================================================================
# ORCHESTRATION
# =====================================================================
def run_pipeline(bands_raw: np.ndarray) -> dict:
    """
    bands_raw: (H, W, 3) float32, channel order Green/Red/NIR, at the
               tile's original resolution. Values can be raw sensor
               range or already-scaled [0,1] -- percentile_normalize()
               handles either, matching the training pipeline's stretch.

    Returns dict with:
        cloud_mask          : (H,W) float32 binary, original resolution
        original_rgb        : (H,W,3) uint8 display image, stage 1 panel
        cloud_removed_rgb    : (H,W,3) uint8 display image, stage 2 panel
        reconstructed_rgb   : (H,W,3) uint8 display image, stage 3 panel
        reconstructed_bands : (H,W,3) float32 [0,1] raw bands, for metrics
    """
    bands_norm = percentile_normalize(bands_raw)
    cloud_mask = detect_cloud_mask(bands_norm)
    masked_vis = visualize_masked_input(bands_norm, cloud_mask)

    input_tensor, original_size = preprocess(bands_norm, cloud_mask)
    model_output = run_model(input_tensor)
    reconstructed_bands = postprocess_output(model_output, original_size)

    return {
        "cloud_mask": cloud_mask,
        "original_rgb": bands_to_display_rgb(bands_norm),
        "cloud_removed_rgb": bands_to_display_rgb(masked_vis),
        "reconstructed_rgb": bands_to_display_rgb(reconstructed_bands),
        "reconstructed_bands": reconstructed_bands,
    }
