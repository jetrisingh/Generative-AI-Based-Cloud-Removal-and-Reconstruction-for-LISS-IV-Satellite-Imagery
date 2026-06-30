# ISRO-CRR — Cloud Removal & Reconstruction for LISS-IV Imagery

A demo web app for the ISRO Hackathon challenge: upload a cloudy LISS-IV
satellite tile, and the pipeline shows the original, the cloud-removed
version, and the final AI-reconstructed version side by side, with metrics.

```
isro-crr/
├── frontend/              static site (HTML/CSS/JS, no build step)
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── assets/presets/    sample demo tiles
└── backend/                FastAPI server
    ├── main.py             API endpoints
    ├── model_pipeline.py   <-- PLUG YOUR MODEL IN HERE
    ├── metrics.py          sharpness / cloud-coverage / PSNR helpers
    └── requirements.txt
```

## 1. Run the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Check it's alive: open `http://localhost:8000/api/health` → `{"status":"ok"}`.

## 2. Run the frontend

The frontend is plain static files — no build tooling needed.

```bash
cd frontend
python3 -m http.server 5500
```

Then open `http://localhost:5500`. `app.js` already points at
`http://localhost:8000` for local dev (see the `API_BASE` constant at the
top of the file).

Alternative: serve both from one process by uncommenting the
`app.mount("/", StaticFiles(...))` line at the bottom of `backend/main.py`,
then just run the backend and visit `http://localhost:8000`.

## 3. The real model is already wired in

`backend/Model.py` contains the actual `CloudRemovalUNet` architecture, and
`backend/model_pipeline.py` is already integrated to match its exact
contract (confirmed from `inference.py` / `ReconsUnet.py` / `DatastePrep.py`):

- Input: `(B, 4, H, W)` float32 `[0,1]` — Green, Red, NIR, then binary cloud
  mask — fixed to 256×256
- Checkpoint format: `{"epoch":..., "net": state_dict, "opt":...}`, loaded
  via `ckpt["net"]`
- Normalization: per-band 2nd/98th percentile stretch (reproduced exactly
  in `percentile_normalize()`) — **not** a simple `/255` division
- Output: `(B, 3, H, W)` float32 `[0,1]` (model ends in Sigmoid)

### To go live with trained weights

1. Get the checkpoint file from your teammate (e.g. `unet_best.pth`,
   produced by `ReconsUnet.py`'s training loop).
2. Drop it at `backend/weights/unet_best.pth`.
3. **Confirm `base_ch`.** `Model.py`'s class default is `64`, but
   `ReconsUnet.py`'s training CLI default is `32` — ask your teammate which
   was actually used (check if they passed `--base_ch` when training). Set
   it via:
   ```bash
   export MODEL_BASE_CH=32   # or 64, or whatever they confirm
   ```
   (Windows PowerShell: `$env:MODEL_BASE_CH="32"`)
   If this is wrong, the backend logs a clear shape-mismatch error on
   startup telling you to check it — it won't fail silently.
4. Install torch (already in `requirements.txt`): `pip install -r requirements.txt`
5. Restart the backend. Look for:
   ```
   [model_pipeline] Loaded CloudRemovalUNet (base_ch=32) from .../unet_best.pth on cpu.
   ```
   instead of the "running DEMO fallback" message.

Until a checkpoint is present, the app runs a lightweight image-processing
stand-in (see the `DEMO fallback` block in `model_pipeline.py`'s
`run_model()`) so you can demo the UI immediately.

### Cloud mask (channel 3)

`detect_cloud_mask()` reproduces `inference.py`'s `auto_cloud_mask()` —
a brightness-based heuristic, dilated. If your teammate has real QGIS-made
cloud masks for specific demo tiles, those will be more accurate; swap the
call site in `run_pipeline()` to load those instead when available.

### Band order for uploads

The model expects strict **Green, Red, NIR** order. `main.py` handles two
upload types:

- **Multiband TIFF** (real LISS-IV tiles) — read via `tifffile`, assumed to
  store exactly 3 raw bands in Green/Red/NIR order. Raw values (not
  pre-scaled) are passed through so `percentile_normalize()` matches the
  training pipeline exactly. If your TIFFs are laid out differently, fix
  the indexing in `_extract_bands_from_tiff()`.
- **Ordinary PNG/JPG** — there's no real NIR in a normal photo, so one is
  approximated from the visible bands. Marked **DEMO ONLY** in the code —
  it's there so you can demo the UI with any image, not to produce results
  you'd present as real model accuracy. For real accuracy demos, upload
  actual multiband TIFF tiles.

### Display mapping

LISS-IV has no blue band, so the app can't show true color. Both panels use
the standard False Color Composite convention (NIR→display-R, Red→display-G,
Green→display-B) — see `bands_to_display_rgb()` in `model_pipeline.py` if
you want a different mapping.


## 4. Metrics shown in the UI

Computed in `backend/metrics.py`, purely from the images your pipeline
returns — no model internals needed:

- **Cloud coverage** — mean of the cloud mask from stage 1
- **Sharpness gain** — relative change in a Laplacian-edge-variance proxy
  between the original and the reconstructed image
- **PSNR (est.)** — signal-to-noise ratio between the cloud-removed and
  reconstructed stages, as an illustrative proxy (swap in a real
  ground-truth comparison if you have held-out clean tiles for validation)

## 5. Deploying for the hackathon

- Any host that runs Python works for the backend (Render, Railway, a VM,
  etc.). Expose port 8000 (or whatever you configure) and update
  `API_BASE` in `frontend/app.js` to that public URL.
- The frontend is static — it can be hosted anywhere (GitHub Pages,
  Netlify, Vercel, or served by the backend itself via the `StaticFiles`
  mount mentioned above).
- Before a public demo, tighten `allow_origins=["*"]` in `main.py`'s CORS
  config to your actual frontend domain.
