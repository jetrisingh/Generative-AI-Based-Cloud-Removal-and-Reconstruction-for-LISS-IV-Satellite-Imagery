// ===================== CONFIG =====================
// Point this at your backend. If frontend and backend are served from the
// same origin (e.g. FastAPI serving these static files too), leave as "".
const API_BASE = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
  ? "http://localhost:8000"
  : ""; // same-origin in production

// Preset demo tiles. Replace these src paths with real sample tiles you ship
// in frontend/assets/presets/. The backend can also serve these from
// /api/presets if you'd rather keep them server-side.
const PRESETS = [
  { id: "liss-iv-001", name: "TERRAIN / VEGETATION", img: "assets/presets/preset1.jpg" },
  { id: "liss-iv-002", name: "URBAN / ROAD",        img: "assets/presets/preset2.jpg" },
  { id: "liss-iv-003", name: "ARID / DESERT",        img: "assets/presets/preset3.jpg" },
];

// ===================== ELEMENTS =====================
const dropzone     = document.getElementById("dropzone");
const fileInput     = document.getElementById("file-input");
const selectBtn     = document.getElementById("select-file-btn");
const presetsGrid   = document.getElementById("presets-grid");
const apiStatusEl   = document.getElementById("api-status");
const toastEl       = document.getElementById("toast");

const panels = {
  original: { ph: document.getElementById("ph-original"), img: document.getElementById("img-original") },
  cloud:    { ph: document.getElementById("ph-cloud"),    img: document.getElementById("img-cloud") },
  recon:    { ph: document.getElementById("ph-recon"),    img: document.getElementById("img-recon") },
};
const metricsRow = document.getElementById("metrics-row");
const mCloud = document.getElementById("m-cloud");
const mSharp = document.getElementById("m-sharp");
const mPsnr  = document.getElementById("m-psnr");
const mTime  = document.getElementById("m-time");

// ===================== INIT =====================
renderPresets();
checkApiHealth();

// ===================== HEALTH CHECK =====================
async function checkApiHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`, { method: "GET" });
    if (!res.ok) throw new Error("bad status");
    apiStatusEl.textContent = "ONLINE";
    apiStatusEl.classList.add("ok");
  } catch (e) {
    apiStatusEl.textContent = "OFFLINE";
    apiStatusEl.classList.add("down");
  }
}

// ===================== PRESETS =====================
function renderPresets() {
  presetsGrid.innerHTML = "";
  PRESETS.forEach((p, i) => {
    const card = document.createElement("div");
    card.className = "preset-card";
    card.innerHTML = `
      <div class="preset-thumb" style="background-image:url('${p.img}')">
        <span class="preset-tag">${p.id.toUpperCase()}</span>
      </div>
      <div class="preset-meta">
        <div class="preset-num">PRESET · ${String(i + 1).padStart(2, "0")}</div>
        <div class="preset-name">${p.name}</div>
      </div>`;
    card.addEventListener("click", () => loadPreset(p));
    presetsGrid.appendChild(card);
  });
}

async function loadPreset(preset) {
  try {
    showToast(`Loading preset ${preset.id}…`);
    const res = await fetch(preset.img);
    const blob = await res.blob();
    const file = new File([blob], `${preset.id}.jpg`, { type: blob.type || "image/jpeg" });
    await runPipeline(file);
  } catch (e) {
    showToast("Could not load preset image.", true);
  }
}

// ===================== DROPZONE / FILE INPUT =====================
selectBtn.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("click", (e) => {
  if (e.target === selectBtn) return;
  fileInput.click();
});

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) runPipeline(file);
});

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) runPipeline(file);
});

// ===================== CORE PIPELINE CALL =====================
async function runPipeline(file) {
  const MAX_BYTES = 25 * 1024 * 1024;
  if (file.size > MAX_BYTES) {
    showToast("File exceeds 25MB limit.", true);
    return;
  }

  resetPanels();
  setPanelLoading("original", "READING FILE…");
  setPanelLoading("cloud", "DEHAZING…");
  setPanelLoading("recon", "RECONSTRUCTING…");

  const form = new FormData();
  form.append("file", file);

  const start = performance.now();
  try {
    const res = await fetch(`${API_BASE}/api/process`, {
      method: "POST",
      body: form,
    });

    if (!res.ok) {
      const errText = await safeText(res);
      throw new Error(errText || `Server returned ${res.status}`);
    }

    const data = await res.json();
    const elapsed = ((performance.now() - start) / 1000).toFixed(2);

    // Expected response shape (see backend/main.py):
    // {
    //   original_image: "data:image/png;base64,....",       // false-color composite of decoded bands
    //   cloud_removed_image: "data:image/png;base64,....",
    //   reconstructed_image: "data:image/png;base64,....",
    //   metrics: { cloud_coverage_pct, sharpness_gain_pct, psnr_db }
    // }
    // Note: we display the server's decoded composite rather than the raw
    // uploaded file, because multiband TIFFs (real LISS-IV tiles) won't
    // render natively in an <img> tag.
    showImage("original", data.original_image);
    showImage("cloud", data.cloud_removed_image);
    showImage("recon", data.reconstructed_image);

    metricsRow.hidden = false;
    mCloud.textContent = formatPct(data.metrics?.cloud_coverage_pct);
    mSharp.textContent = formatPct(data.metrics?.sharpness_gain_pct, true);
    mPsnr.textContent  = data.metrics?.psnr_db != null ? `${data.metrics.psnr_db} dB` : "—";
    mTime.textContent  = `${elapsed}s`;

  } catch (err) {
    console.error(err);
    setPanelError("original", "READ FAILED");
    setPanelError("cloud", "PROCESSING FAILED");
    setPanelError("recon", "PROCESSING FAILED");
    showToast(`Pipeline error: ${err.message}`, true);
  }
}

// ===================== PANEL HELPERS =====================
function resetPanels() {
  Object.values(panels).forEach(({ ph, img }) => {
    img.hidden = true;
    img.src = "";
    ph.hidden = false;
  });
  metricsRow.hidden = true;
}

function setPanelLoading(key, label) {
  panels[key].ph.innerHTML = `${label}<br/><span class="dim">PROCESSING…</span>`;
}

function setPanelError(key, label) {
  panels[key].img.hidden = true;
  panels[key].ph.hidden = false;
  panels[key].ph.innerHTML = `${label}<br/><span class="dim">SEE TOAST FOR DETAILS</span>`;
}

function showImage(key, src) {
  panels[key].ph.hidden = true;
  panels[key].img.hidden = false;
  panels[key].img.src = src;
}

function formatPct(v, signed = false) {
  if (v == null) return "—";
  const sign = signed && v > 0 ? "+" : "";
  return `${sign}${v}%`;
}

async function safeText(res) {
  try {
    const j = await res.json();
    return j.detail || j.message || JSON.stringify(j);
  } catch {
    return res.statusText;
  }
}

// ===================== TOAST =====================
let toastTimer = null;
function showToast(msg, isError = false) {
  clearTimeout(toastTimer);
  toastEl.textContent = msg;
  toastEl.classList.toggle("error", isError);
  toastEl.hidden = false;
  toastTimer = setTimeout(() => (toastEl.hidden = true), 4000);
}
