/* ===================== Theme handling ===================== */
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("ff-theme", theme);
  document.querySelectorAll("[data-theme-choice]").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.themeChoice === theme);
  });
}
function initTheme() {
  const saved = localStorage.getItem("ff-theme") || "light";
  applyTheme(saved);
  document.querySelectorAll("[data-theme-choice]").forEach((btn) => {
    btn.addEventListener("click", () => applyTheme(btn.dataset.themeChoice));
  });
}

/* ===================== Global state ===================== */
let activeAnalysis = null; // see buildAnalysisRecord() for shape
let analysisCounter = 123;
let availableModels = [];
let analysisHistory = []; // newest first - persisted to localStorage, survives reloads (no backend DB yet)

const HISTORY_STORAGE_KEY = "ff_analysis_history";
const COUNTER_STORAGE_KEY = "ff_analysis_counter";
const MAX_HISTORY_ENTRIES = 15; // caps localStorage size (entries embed base64 images)

function loadPersistedHistory() {
  try {
    const saved = JSON.parse(localStorage.getItem(HISTORY_STORAGE_KEY) || "[]");
    analysisHistory = Array.isArray(saved) ? saved : [];
  } catch (e) {
    analysisHistory = [];
  }
  const savedCounter = Number(localStorage.getItem(COUNTER_STORAGE_KEY));
  if (!Number.isNaN(savedCounter) && savedCounter > analysisCounter) analysisCounter = savedCounter;
}

function addToHistory(record) {
  analysisHistory.unshift(record);
  analysisHistory = analysisHistory.slice(0, MAX_HISTORY_ENTRIES);
  try {
    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(analysisHistory));
    localStorage.setItem(COUNTER_STORAGE_KEY, String(analysisCounter));
  } catch (e) {
    // Quota exceeded - drop older half and retry once, then give up silently
    // (in-memory history still works for the rest of this session either way).
    try {
      analysisHistory = analysisHistory.slice(0, Math.floor(MAX_HISTORY_ENTRIES / 2));
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(analysisHistory));
    } catch (e2) { /* storage unavailable - continue in-memory only */ }
  }
}

function nextAnalysisId() {
  analysisCounter += 1;
  return `FA-${String(analysisCounter).padStart(6, "0")}`;
}

/* ===================== Navigation ===================== */
function initNav() {
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => goToPage(btn.dataset.page));
  });
}

function goToPage(pageKey) {
  document.querySelectorAll(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.page === pageKey));
  document.querySelectorAll(".page").forEach((p) => p.classList.remove("active"));
  document.getElementById(`page-${pageKey}`).classList.add("active");

  if (pageKey === "gradcam") renderGradcamPage();
  if (pageKey === "report") renderReportPage();
  if (pageKey === "history") renderHistoryPage();
  if (pageKey === "benchmark") renderBenchmarkPage();
  if (pageKey === "guide") renderGuidePage();
}

/* ===================== Health check ===================== */
let mainPageModelKey = "noaug"; // Model 1 = noaug (production default), Model 2 = best (research checkpoint, local only)

async function checkHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    availableModels = data.loaded_models || [];
  } catch (e) {
    availableModels = [];
  }
  updateModelStatus("main", mainPageModelKey, "status-main");
  updateModelStatus("crossdomain", "cross_domain", "status-crossdomain");
}

function initMainModelToggle() {
  const toggle = document.getElementById("main-model-toggle");
  if (!toggle) return;
  toggle.querySelectorAll(".model-toggle-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const modelKey = btn.dataset.mainModel;
      const modelLabel = btn.dataset.mainLabel;
      mainPageModelKey = modelKey;

      toggle.querySelectorAll(".model-toggle-btn").forEach((b) => b.classList.toggle("active", b === btn));

      const pageMain = document.getElementById("page-main");
      pageMain.dataset.modelKey = modelKey;
      pageMain.dataset.modelLabel = modelLabel;

      const body = document.querySelector('[data-page-body="main"]');
      if (body) body.dataset.locked = ""; // allow re-render even if a previous analysis is mid-flight

      updateModelStatus("main", modelKey, "status-main");
    });
  });
}

/* ===================== Model details modal ===================== */
const MODEL_DETAILS = {
  noaug: {
    heading: "Model 1 — No-Augmentation Model",
    subheading: "mobilenetv3_noaug.pth · Production default",
    rows: [
      ["Checkpoint", "mobilenetv3_noaug.pth"],
      ["Architecture", "MobileNetV3-Large (ImageNet-pretrained backbone, custom classifier head)"],
      ["Training strategy", "2-stage transfer learning, no data augmentation applied during training"],
      ["Training data", "FFHQ (real) + Stable Diffusion (fake), no augmentation transforms"],
      ["Held-out test accuracy", "95.98% (2,401 images)"],
      ["Fake recall", "100% (0 missed fakes)"],
      ["Real false alarms", "96 real images flagged fake"],
      ["Deployment role", "Production default — deployed on Render. Chosen over Model 2 despite lower accuracy because it misses zero fakes, which matters more than false alarms in security-critical screening."],
    ],
  },
  best: {
    heading: "Model 2 — Research / Best Model",
    subheading: "mobilenetv3_best.pth · Deployed, opt-in via this toggle",
    rows: [
      ["Checkpoint", "mobilenetv3_best.pth"],
      ["Architecture", "MobileNetV3-Large (same backbone/head as Model 1)"],
      ["Training strategy", "3-stage progressive fine-tuning (frozen backbone → partial unfreeze → full unfreeze)"],
      ["Training data", "FFHQ (15,000 real) + Stable Diffusion (9,001 fake) + CelebA-HD (8,000 real, added in Stage 3 to fix smartphone-photo false positives)"],
      ["Held-out test accuracy", "99.63% (2,401 images)"],
      ["Fake recall", "99.89% (1 missed fake)"],
      ["Real false alarms", "8 real images flagged fake"],
      ["Deployment role", "Research/comparison checkpoint — deployed on Render alongside Model 1, selectable via this toggle. Higher overall accuracy than Model 1, but not the default because it occasionally misses a fake (Model 1 misses zero)."],
    ],
  },
};

function initModelDetailsModal() {
  const overlay = document.getElementById("model-details-modal");
  const content = document.getElementById("model-details-content");
  const closeBtn = document.getElementById("model-details-close");
  if (!overlay || !content || !closeBtn) return;

  function openModal(modelKey) {
    const info = MODEL_DETAILS[modelKey];
    if (!info) return;
    const rowsHtml = info.rows
      .map(([label, value]) => `<tr><td>${label}</td><td>${value}</td></tr>`)
      .join("");
    content.innerHTML = `
      <h2 class="modal-heading" id="model-details-heading">${info.heading}</h2>
      <p class="modal-subheading">${info.subheading}</p>
      <table class="modal-table">${rowsHtml}</table>
    `;
    overlay.classList.add("open");
  }

  function closeModal() {
    overlay.classList.remove("open");
  }

  document.querySelectorAll(".model-toggle-details-link").forEach((link) => {
    link.addEventListener("click", () => openModal(link.dataset.detailsModel));
  });

  closeBtn.addEventListener("click", closeModal);
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) closeModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && overlay.classList.contains("open")) closeModal();
  });
}

function updateModelStatus(pageKey, modelKey, statusElId) {
  const el = document.getElementById(statusElId);
  const available = availableModels.includes(modelKey);
  if (available) {
    el.innerHTML = `<span class="status-dot"></span> Model Available`;
    el.classList.remove("unavailable");
  } else {
    el.innerHTML = `<span class="status-dot"></span> Model not loaded on server yet — checkpoint unavailable`;
    el.classList.add("unavailable");
  }
  renderStageForModel(pageKey, modelKey, available);
}

/* ===================== Format helpers ===================== */
function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}
function fileTypeLabel(file) {
  const t = (file.type || "").split("/")[1];
  return t ? t.toUpperCase() : (file.name.split(".").pop() || "?").toUpperCase();
}
const PREVIEW_MAX_DIM = 1280; // longest side, px
const PREVIEW_JPEG_QUALITY = 0.82;
const UPLOAD_MAX_DIM = 1600; // matches preprocessing.py's MAX_DETECTION_DIM -
// the server downscales to this same cap before face detection anyway, and
// the model only ever sees a 224x224 crop, so resizing here loses nothing
// but cuts upload payload size and server-side decode time for large
// photos (this is what actually gets sent to /predict, not just the preview).
const UPLOAD_JPEG_QUALITY = 0.9;

function readImageDims(file) {
  // Renders a size-capped JPEG data: URL for the preview/history/report
  // pipeline instead of embedding the raw original file. Uploads in the
  // 5-10MB range (phone photos are often 3000-4000px) were producing
  // multi-megabyte base64 strings that stalled the browser decoding/
  // painting the preview and then blew past localStorage's per-origin
  // quota when saved to history. A data: URL (not URL.createObjectURL) is
  // still used so it survives JSON-serializing into localStorage and
  // still works after a page reload - blob: URLs are invalidated as soon
  // as the page that created them unloads.
  //
  // Also produces `uploadBlob`: the actual payload sent to /predict,
  // downscaled to UPLOAD_MAX_DIM. A >4-6MB phone photo previously went to
  // the server untouched, which combined with Render's 512MB free-tier RAM
  // caused request timeouts/crashes during preprocessing. If the original
  // is already <= UPLOAD_MAX_DIM, the original file is uploaded unchanged
  // (no needless re-encode).
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        const w = img.naturalWidth;
        const h = img.naturalHeight;
        const longest = Math.max(w, h);

        const previewScale = longest > PREVIEW_MAX_DIM ? PREVIEW_MAX_DIM / longest : 1;
        const previewCanvas = document.createElement("canvas");
        previewCanvas.width = Math.max(1, Math.round(w * previewScale));
        previewCanvas.height = Math.max(1, Math.round(h * previewScale));
        previewCanvas.getContext("2d").drawImage(img, 0, 0, previewCanvas.width, previewCanvas.height);
        const url = previewCanvas.toDataURL("image/jpeg", PREVIEW_JPEG_QUALITY);

        if (longest <= UPLOAD_MAX_DIM) {
          resolve({ w, h, url, uploadBlob: file });
          return;
        }
        const uploadScale = UPLOAD_MAX_DIM / longest;
        const uploadCanvas = document.createElement("canvas");
        uploadCanvas.width = Math.max(1, Math.round(w * uploadScale));
        uploadCanvas.height = Math.max(1, Math.round(h * uploadScale));
        uploadCanvas.getContext("2d").drawImage(img, 0, 0, uploadCanvas.width, uploadCanvas.height);
        uploadCanvas.toBlob(
          (blob) => resolve({ w, h, url, uploadBlob: blob || file }),
          "image/jpeg",
          UPLOAD_JPEG_QUALITY
        );
      };
      img.onerror = () => reject(new Error("Could not read this image file."));
      img.src = reader.result;
    };
    reader.onerror = () => reject(new Error("Could not read this image file."));
    reader.readAsDataURL(file);
  });
}
function nowString() {
  const d = new Date();
  return d.toLocaleDateString(undefined, { day: "2-digit", month: "long", year: "numeric" }) +
    ", " + d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

/* ===================== Upload stage (Main + Cross-Domain pages share this) ===================== */
// Keyed by "pageKey:modelKey" (not just pageKey) so the Main page's Model
// 1/Model 2 toggle keeps each model's uploaded image/result independent -
// switching the toggle used to wipe whatever was there, even switching
// back to a model you'd already analyzed on.
function stateKey(pageKey, modelKey) {
  return `${pageKey}:${modelKey}`;
}
const pageFileState = {}; // stateKey -> { file, previewUrl, w, h, uploadBlob }
const pageResults = {}; // stateKey -> completed analysis record (same shape as activeAnalysis)

// Sample gallery: pre-loaded real/fake images so evaluators can try the
// model without needing their own test images. Files live in
// static/samples/<category>/<n>.jpg (see webapp/backend/prepare_samples.py).
const SAMPLE_SETS = {
  main: { real: { category: "main-real", count: 10 }, fake: { category: "main-fake", count: 10 } },
  crossdomain: { real: { category: "cross-real", count: 10 }, fake: { category: "cross-fake", count: 9 } },
};

function sampleGalleryHtml(pageKey) {
  const set = SAMPLE_SETS[pageKey];
  if (!set) return "";
  const thumb = (category, n) =>
    `<button type="button" class="sample-thumb" data-sample-category="${category}" data-sample-n="${n}" title="Sample ${n}">
      <img src="samples/${category}/${n}.jpg" alt="${category} sample ${n}" loading="lazy" />
    </button>`;
  const realThumbs = Array.from({ length: set.real.count }, (_, i) => thumb(set.real.category, i + 1)).join("");
  const fakeThumbs = Array.from({ length: set.fake.count }, (_, i) => thumb(set.fake.category, i + 1)).join("");

  return `
    <div class="sample-gallery">
      <button type="button" class="sample-gallery-toggle" id="sample-toggle-${pageKey}">
        &#9656; Try a sample image
      </button>
      <div class="sample-gallery-body" id="sample-body-${pageKey}" hidden>
        <div class="sample-gallery-label">Real (${set.real.count})</div>
        <div class="sample-thumb-row">${realThumbs}</div>
        <div class="sample-gallery-label">AI-Generated / Fake (${set.fake.count})</div>
        <div class="sample-thumb-row">${fakeThumbs}</div>
      </div>
    </div>
  `;
}

function wireSampleGallery(pageKey, modelKey, modelAvailable) {
  const toggle = document.getElementById(`sample-toggle-${pageKey}`);
  const body = document.getElementById(`sample-body-${pageKey}`);
  if (!toggle || !body) return;

  toggle.addEventListener("click", () => {
    const isHidden = body.hasAttribute("hidden");
    if (isHidden) body.removeAttribute("hidden"); else body.setAttribute("hidden", "");
    toggle.innerHTML = (isHidden ? "&#9662; Try a sample image" : "&#9656; Try a sample image");
  });

  body.querySelectorAll(".sample-thumb").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const category = btn.dataset.sampleCategory;
      const n = btn.dataset.sampleN;
      body.querySelectorAll(".sample-thumb").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      try {
        const res = await fetch(`samples/${category}/${n}.jpg`);
        const blob = await res.blob();
        const file = new File([blob], `${category}-${n}.jpg`, { type: "image/jpeg" });
        handleFileSelected(pageKey, modelKey, file, modelAvailable);
      } catch (e) {
        showUploadError(pageKey, "Could not load the sample image. Try again or upload your own.");
      }
    });
  });
}

// Dispatches to the right stage for this (page, model) pair - restores a
// previously-completed result or in-progress upload for this model
// instead of always resetting to a blank dropzone.
function renderStageForModel(pageKey, modelKey, modelAvailable) {
  const container = document.querySelector(`[data-page-body="${pageKey}"]`);
  if (container && container.dataset.locked === "1") return;

  const key = stateKey(pageKey, modelKey);
  if (!pageFileState[key]) {
    renderUploadStage(pageKey, modelKey, modelAvailable);
    return;
  }
  renderPreviewStage(pageKey, modelKey, modelAvailable);
  if (pageResults[key]) {
    activeAnalysis = pageResults[key];
    const resultArea = document.getElementById(`result-area-${pageKey}`);
    if (resultArea) renderResultCard(pageKey, resultArea);
  }
}

function renderUploadStage(pageKey, modelKey, modelAvailable) {
  const container = document.querySelector(`[data-page-body="${pageKey}"]`);
  if (!container) return;
  // Don't clobber an in-progress analyzing/result view on re-check of health
  if (container.dataset.locked === "1") return;

  container.innerHTML = `
    <div class="upload-grid">
      <div class="dropzone" id="dropzone-${pageKey}">
        <div class="dropzone-icon">
          <svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M12 16V4M12 4L7 9M12 4l5 5" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" stroke-linecap="round"/>
          </svg>
        </div>
        <div class="dropzone-title">UPLOAD IMAGE</div>
        <div class="dropzone-sub">Drag &amp; drop your image here</div>
        <div class="dropzone-sub">or</div>
        <button type="button" class="browse-btn">Browse from Computer</button>
        <input type="file" id="file-input-${pageKey}" accept="image/jpeg,image/png,image/jpg" hidden />
        <div class="dropzone-formats">JPG &nbsp;&bull;&nbsp; JPEG &nbsp;&bull;&nbsp; PNG</div>
        <div class="dropzone-maxsize">Maximum file size: 10 MB</div>
      </div>
      <div class="preview-card" style="align-items:center; justify-content:center; color:var(--muted); text-align:center;">
        Upload an image to begin analysis.
      </div>
    </div>
    ${sampleGalleryHtml(pageKey)}
  `;

  const dropzone = document.getElementById(`dropzone-${pageKey}`);
  const fileInput = document.getElementById(`file-input-${pageKey}`);

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length) handleFileSelected(pageKey, modelKey, e.dataTransfer.files[0], modelAvailable);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFileSelected(pageKey, modelKey, fileInput.files[0], modelAvailable);
  });

  wireSampleGallery(pageKey, modelKey, modelAvailable);
}

const MAX_UPLOAD_BYTES = 10 * 1024 * 1024; // matches config.MAX_UPLOAD_BYTES server-side

function showUploadError(pageKey, message) {
  const slot = document.querySelector(`[data-page-body="${pageKey}"] .preview-card`);
  if (slot) slot.innerHTML = `<div style="color:var(--fake); text-align:center; padding:20px;">${message}</div>`;
}

async function handleFileSelected(pageKey, modelKey, file, modelAvailable) {
  if (!file.type.startsWith("image/")) return;
  if (file.size > MAX_UPLOAD_BYTES) {
    showUploadError(pageKey, `"${file.name}" is ${formatBytes(file.size)}, over the 10 MB limit. Choose a smaller image.`);
    return;
  }
  try {
    const dims = await readImageDims(file);
    pageFileState[stateKey(pageKey, modelKey)] = { file, previewUrl: dims.url, w: dims.w, h: dims.h, uploadBlob: dims.uploadBlob };
    delete pageResults[stateKey(pageKey, modelKey)]; // a fresh upload invalidates any earlier result for this model
    renderPreviewStage(pageKey, modelKey, modelAvailable);
  } catch (e) {
    showUploadError(pageKey, "Could not read this image file. Try a different file.");
  }
}

function renderPreviewStage(pageKey, modelKey, modelAvailable) {
  const container = document.querySelector(`[data-page-body="${pageKey}"]`);
  const { file, previewUrl, w, h } = pageFileState[stateKey(pageKey, modelKey)];

  container.innerHTML = `
    <div class="upload-grid">
      <div class="dropzone" id="dropzone-${pageKey}" style="min-height:auto; padding:16px;">
        <div class="dropzone-sub">Upload a different image to replace this one.</div>
        <button type="button" class="browse-btn">Browse from Computer</button>
        <input type="file" id="file-input-${pageKey}" accept="image/jpeg,image/png,image/jpg" hidden />
      </div>
      <div class="preview-card">
        <img class="preview-img" src="${previewUrl}" alt="preview" />
        <div class="preview-meta">
          <div class="preview-meta-row">&#128196; ${file.name}</div>
          <div class="preview-meta-row">&#128444; ${w} &times; ${h}</div>
          <div class="preview-meta-row">&#128190; ${formatBytes(file.size)}</div>
          <div class="preview-actions">
            <button type="button" class="btn btn-block" id="replace-btn-${pageKey}">Replace Image</button>
            <button type="button" class="btn btn-primary btn-block" id="analyze-btn-${pageKey}" ${modelAvailable ? "" : "disabled"}>
              Analyze Image
            </button>
          </div>
        </div>
      </div>
    </div>
    <div id="result-area-${pageKey}" style="margin-top:20px;"></div>
  `;

  const dropzone = document.getElementById(`dropzone-${pageKey}`);
  const fileInput = document.getElementById(`file-input-${pageKey}`);
  dropzone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFileSelected(pageKey, modelKey, fileInput.files[0], modelAvailable);
  });
  document.getElementById(`replace-btn-${pageKey}`).addEventListener("click", () => fileInput.click());

  const analyzeBtn = document.getElementById(`analyze-btn-${pageKey}`);
  if (!modelAvailable) {
    document.getElementById(`result-area-${pageKey}`).innerHTML =
      `<p style="color:var(--fake); font-size:0.85rem;">This model's checkpoint isn't loaded on the server yet, so analysis can't run.</p>`;
  } else {
    analyzeBtn.addEventListener("click", () => runAnalysis(pageKey, modelKey));
  }
}

async function runAnalysis(pageKey, modelKey) {
  const container = document.querySelector(`[data-page-body="${pageKey}"]`);
  container.dataset.locked = "1";
  const resultArea = document.getElementById(`result-area-${pageKey}`);
  const analyzeBtn = document.getElementById(`analyze-btn-${pageKey}`);
  if (analyzeBtn) analyzeBtn.disabled = true;
  // Captured NOW, synchronously, before any await - if we read this after
  // the fetch resolves instead, a toggle switch mid-flight (Model 1 -> 2)
  // would read Model 2's label and mislabel Model 1's result.
  const modelLabel = document.getElementById(`page-${pageKey}`).dataset.modelLabel;

  resultArea.innerHTML = `
    <div class="card analyzing-card">
      <div class="spinner-ring"></div>
      <div class="analyzing-title">ANALYZING IMAGE</div>
      <div class="analyzing-sub">Processing the submitted image…</div>
      <div class="step-list">
        <div>&#10003; Image validation</div>
        <div>&#10003; Face detection</div>
        <div>&#10003; Preprocessing</div>
        <div>&#10003; Model inference</div>
        <div>&#10003; Confidence calculation</div>
        <div>&hellip; Forensic scan (meta-detector)</div>
      </div>
    </div>
  `;

  const { file, previewUrl, w, h, uploadBlob } = pageFileState[stateKey(pageKey, modelKey)];
  const formData = new FormData();
  formData.append("file", uploadBlob || file, file.name);

  try {
    const res = await fetch(`/predict?model=${modelKey}`, { method: "POST", body: formData });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    const p = data.prediction;

    const result = {
      analysisId: nextAnalysisId(),
      modelKey, modelLabel, modelVersion: "v1.0",
      pageKey,
      filename: file.name,
      fileType: fileTypeLabel(file),
      resolution: `${w} × ${h}`,
      fileSize: formatBytes(file.size),
      colorMode: "RGB (converted for inference)",
      previewUrl,
      label: p.label,
      realPct: p.real_pct,
      fakePct: p.fake_pct,
      heatmapB64: null,
      overlayB64: null,
      uploadBlob: uploadBlob || file,
      faceAlignmentUsed: data.face_alignment_used,
      meta: data.meta_detector || null,
      generatedAt: nowString(),
    };

    // Always save the result, even if the user has since switched away from
    // this model - renderStageForModel() will pick it up next time they
    // switch back. Only touch the shared result-area DOM (and activeAnalysis)
    // if this model is still the one actually on screen; otherwise those
    // elements may have already been replaced by the model switched to.
    pageResults[stateKey(pageKey, modelKey)] = result;
    addToHistory({ ...result, status: "completed" });
    if (document.getElementById(`page-${pageKey}`).dataset.modelKey === modelKey) {
      activeAnalysis = result;
      renderResultCard(pageKey, resultArea);
    }
  } catch (e) {
    addToHistory({
      analysisId: nextAnalysisId(), modelKey, modelLabel,
      filename: file.name, resolution: `${w} × ${h}`, previewUrl,
      status: "failed", errorMessage: e.message, generatedAt: nowString(),
    });
    if (document.getElementById(`page-${pageKey}`).dataset.modelKey === modelKey) {
      resultArea.innerHTML = `<p style="color:var(--fake);">Analysis failed: ${e.message}</p>`;
    }
  } finally {
    // Don't clear the lock/button state for a model that's no longer being
    // displayed - it would incorrectly unlock whatever model the user has
    // since switched to (which has its own lock state from its own click).
    if (document.getElementById(`page-${pageKey}`).dataset.modelKey === modelKey) {
      container.dataset.locked = "";
      if (analyzeBtn) analyzeBtn.disabled = false;
    }
  }
}

function renderResultCard(pageKey, resultArea) {
  const a = activeAnalysis;
  const isReal = a.label === "Real";
  const verdictText = isReal ? "REAL / AUTHENTIC" : "AI GENERATED";
  const confidence = isReal ? a.realPct : a.fakePct;

  resultArea.innerHTML = `
    <h2 style="font-size:1rem; letter-spacing:0.05em; margin-bottom:14px;">AUTHENTICITY RESULT</h2>
    <div class="result-grid">
      <div>
        <div class="verdict-box ${isReal ? "real" : "fake"}">
          <div class="verdict-icon">${isReal ? "&#9989;" : "&#9888;&#65039;"}</div>
          <div>
            <div class="verdict-label">${verdictText}</div>
            <div class="verdict-conf">${confidence.toFixed(2)}%</div>
            <div class="verdict-conf-label">Confidence Score</div>
          </div>
        </div>
        <div class="metric-tiles">
          <div class="metric-tile ai">
            <div class="metric-tile-label">AI PROBABILITY</div>
            <div class="metric-tile-value">${a.fakePct.toFixed(2)}%</div>
            <div class="bar-track"><div class="bar-fill ai" style="width:${a.fakePct}%"></div></div>
          </div>
          <div class="metric-tile real">
            <div class="metric-tile-label">REAL PROBABILITY</div>
            <div class="metric-tile-value">${a.realPct.toFixed(2)}%</div>
            <div class="bar-track"><div class="bar-fill real" style="width:${a.realPct}%"></div></div>
          </div>
        </div>
      </div>
      <div class="info-panel">
        <div class="info-panel-title">MODEL INFORMATION</div>
        <div class="info-row"><span class="k">Model</span><span>${a.modelLabel}</span></div>
        <div class="info-row"><span class="k">Version</span><span>${a.modelVersion}</span></div>
        <div class="info-row"><span class="k">Status</span><span class="status-completed">Completed</span></div>
        <div class="info-row"><span class="k">Analysis ID</span><span>${a.analysisId}</span></div>
      </div>
    </div>
    <div class="action-row">
      <button class="btn" id="view-gradcam-btn-${pageKey}">&#8857; Run Grad-CAM</button>
      <button class="btn btn-primary" id="generate-report-btn-${pageKey}">&#128196; Generate Forensic Report</button>
    </div>
    ${metaPanelHtml(a)}
  `;

  resultArea.querySelector(`#view-gradcam-btn-${pageKey}`).addEventListener("click", () => goToPage("gradcam"));
  resultArea.querySelector(`#generate-report-btn-${pageKey}`).addEventListener("click", () => goToPage("report"));
}

function metaVerdictClass(verdict) {
  const v = (verdict || "").toLowerCase();
  if (v.includes("ai")) return "fake";
  if (v.includes("real") || v.includes("photograph")) return "real";
  return "uncertain";
}

function metaPanelHtml(a) {
  const m = a.meta;
  if (!m) return "";
  const md = (m.signals && m.signals.metadata) || {};
  const wm = (m.signals && m.signals.watermark) || {};
  const freq = (m.signals && m.signals.frequency) || {};
  const noise = (m.signals && m.signals.noise) || {};
  const ela = (m.signals && m.signals.ela) || {};

  const flags = [];
  if (wm.found) flags.push(`<span class="badge badge-meta-fake">&#9888; Watermark: ${wm.label || wm.message}</span>`);
  if (md.c2pa && md.c2pa.length) flags.push(`<span class="badge badge-meta">&#128278; C2PA</span>`);
  if (md.synthetic_source_tag) flags.push(`<span class="badge badge-meta-fake">&#9888; Synthetic source tag</span>`);
  if (md.camera_markers && md.camera_markers.length) flags.push(`<span class="badge badge-meta-real">&#128247; Camera metadata</span>`);
  if (md.has_gps) flags.push(`<span class="badge badge-meta-real">&#128506; GPS</span>`);
  if (md.has_datetime) flags.push(`<span class="badge badge-meta-real">&#128337; Timestamp</span>`);
  if (md.png_text_keys && md.png_text_keys.length) flags.push(`<span class="badge badge-meta">&#128221; PNG text chunks</span>`);

  const signalTiles = [
    ["AI SCORE", m.ai_score != null ? `${(m.ai_score * 100).toFixed(1)}%` : "—"],
    ["HF ENERGY", freq.high_frequency_ratio != null ? freq.high_frequency_ratio.toFixed(3) : "—"],
    ["SHOT-NOISE CORR", noise.shot_noise_correlation != null ? noise.shot_noise_correlation.toFixed(3) : "—"],
    ["DOUBLE JPEG", freq.double_jpeg_score != null ? freq.double_jpeg_score.toFixed(2) : "—"],
    ["NOISE LEVEL", noise.noise_level != null ? noise.noise_level.toFixed(2) : "—"],
    ["ELA SPREAD", ela.ela_block_cv != null ? ela.ela_block_cv.toFixed(3) : "—"],
  ].map(([k, v]) => `
    <div class="meta-signal-tile">
      <div class="metric-tile-label">${k}</div>
      <div class="meta-signal-value">${v}</div>
    </div>`).join("");

  const evidence = (m.evidence && m.evidence.length)
    ? m.evidence.map((e) => `<li>${e}</li>`).join("")
    : `<li style="color:var(--muted);">No strong forensic signals detected by the meta-detector.</li>`;

  const warnings = (m.warnings && m.warnings.length)
    ? `<div class="meta-warnings">${m.warnings.map((w) => `<div>&#9888; ${w}</div>`).join("")}</div>` : "";

  return `
    <h2 style="font-size:1rem; letter-spacing:0.05em; margin:26px 0 14px;">FORENSIC SCAN &mdash; META DETECTOR</h2>
    <div class="meta-panel">
      <div class="meta-panel-top">
        <div class="verdict-box ${metaVerdictClass(m.verdict)}" style="padding:18px 22px; flex:1;">
          <div class="verdict-icon">${metaVerdictClass(m.verdict) === "real" ? "&#9989;" : "&#128269;"}</div>
          <div>
            <div class="verdict-label" style="font-size:1.05rem;">${m.verdict}</div>
            <div class="verdict-conf" style="font-size:1.5rem;">${m.ai_score != null ? (m.ai_score * 100).toFixed(1) : "—"}%</div>
            <div class="verdict-conf-label">Meta-Detector AI Score</div>
          </div>
        </div>
        <div class="meta-panel-notes">
          <div class="info-panel-title">DETECTED SIGNALS</div>
          <div class="meta-flags">${flags.length ? flags.join("") : `<span style="color:var(--muted); font-size:0.85rem;">No notable markers detected</span>`}</div>
          ${warnings}
        </div>
      </div>
      <div class="meta-signal-grid">${signalTiles}</div>
      <div class="meta-evidence">
        <div class="info-panel-title">EVIDENCE</div>
        <ul>${evidence}</ul>
      </div>
    </div>
  `;
}

/* ===================== Grad-CAM page ===================== */
let gradcamMode = "overlay";
let gradcamMethod = "gradcam"; // "gradcam" | "layercam" | "layercam_fused" - experimental, compares CAM variants
let gradcamIntensity = 65;

function emptyStateHtml(title, sub) {
  return `
    <div class="empty-state">
      <div class="empty-state-icon">&#8857;</div>
      <div class="empty-state-title">${title}</div>
      <div class="empty-state-sub">${sub}</div>
      <button class="btn btn-primary" onclick="goToPage('main')">Go to Main Model</button>
    </div>
  `;
}

// history entries are JSON-persisted to localStorage, which can't hold a
// real Blob (a.uploadBlob survives only within the current page session) -
// on a reopened history entry from a previous session, fall back to
// re-deriving the upload bytes from the already-persisted preview data URL.
async function resolveUploadBlob(a) {
  if (a.uploadBlob instanceof Blob) return a.uploadBlob;
  const res = await fetch(a.previewUrl);
  return res.blob();
}

async function renderGradcamPage() {
  const body = document.getElementById("gradcam-body");
  if (!activeAnalysis) {
    body.innerHTML = emptyStateHtml(
      "NO ANALYSIS AVAILABLE",
      "Complete an image analysis first to generate a Grad-CAM explanation."
    );
    return;
  }
  const a = activeAnalysis;

  if (!a.heatmapB64) {
    body.innerHTML = `
      <div class="card analyzing-card">
        <div class="spinner-ring"></div>
        <div class="analyzing-title">RUNNING GRAD-CAM</div>
        <div class="analyzing-sub">Computing the model attention heatmap…</div>
      </div>
    `;
    try {
      const formData = new FormData();
      formData.append("file", await resolveUploadBlob(a), a.filename);
      const res = await fetch(`/gradcam?model=${a.modelKey}`, { method: "POST", body: formData });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      a.heatmapB64 = data.gradcam_heatmap;
      a.overlayB64 = data.gradcam_overlay;
    } catch (e) {
      body.innerHTML = `<p style="color:var(--fake); padding:16px;">Grad-CAM failed: ${e.message}</p>`;
      return;
    }
  }

  renderGradcamVisualization(body);
}

const CAM_METHOD_LABELS = {
  gradcam: "Grad-CAM",
  layercam: "Layer-CAM",
  layercam_fused: "Layer-CAM (Fused)",
};

function currentCamResult(a) {
  if (gradcamMethod === "gradcam") return { heatmap: a.heatmapB64, overlay: a.overlayB64 };
  const cached = a.camCache && a.camCache[gradcamMethod];
  return cached || { heatmap: a.heatmapB64, overlay: a.overlayB64 };
}

async function switchCamMethod(method, body) {
  const a = activeAnalysis;
  gradcamMethod = method;
  a.camCache = a.camCache || {};

  if (method !== "gradcam" && !a.camCache[method]) {
    body.innerHTML = `
      <div class="card analyzing-card">
        <div class="spinner-ring"></div>
        <div class="analyzing-title">COMPUTING ${CAM_METHOD_LABELS[method].toUpperCase()}</div>
        <div class="analyzing-sub">Running the alternate explainability method…</div>
      </div>
    `;
    try {
      const formData = new FormData();
      formData.append("file", await resolveUploadBlob(a), a.filename);
      const res = await fetch(`/gradcam?model=${a.modelKey}&cam_method=${method}`, { method: "POST", body: formData });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      a.camCache[method] = { heatmap: data.gradcam_heatmap, overlay: data.gradcam_overlay };
    } catch (e) {
      body.innerHTML = `<p style="color:var(--fake); padding:16px;">${CAM_METHOD_LABELS[method]} failed: ${e.message}</p>`;
      return;
    }
  }

  renderGradcamVisualization(body);
}

function renderGradcamVisualization(body) {
  const a = activeAnalysis;
  const isReal = a.label === "Real";
  const cam = currentCamResult(a);

  body.innerHTML = `
    <div class="gradcam-meta-bar">
      <div class="gradcam-meta-item"><div class="k">ANALYSIS ID</div><div class="v">${a.analysisId}</div></div>
      <div class="gradcam-meta-item"><div class="k">MODEL</div><div class="v">${a.modelLabel}</div></div>
      <div class="gradcam-meta-item"><div class="k">PREDICTION</div><div class="v" style="color:${isReal ? "var(--real)" : "var(--fake)"}">${isReal ? "Real" : "AI Generated"}</div></div>
      <div class="gradcam-meta-item"><div class="k">CONFIDENCE</div><div class="v">${(isReal ? a.realPct : a.fakePct).toFixed(2)}%</div></div>
    </div>

    <div class="gradcam-controls" style="margin-bottom:12px;">
      <div class="mode-toggle" id="cam-method-toggle">
        <button class="mode-btn" data-method="gradcam">Grad-CAM</button>
        <button class="mode-btn" data-method="layercam">Layer-CAM</button>
        <button class="mode-btn" data-method="layercam_fused">Layer-CAM (Fused)</button>
      </div>
    </div>
    <p style="font-size:0.78rem; color:var(--muted); margin:-6px 0 14px 0;">Experimental — compares different explainability methods on the same prediction. Not a forgery localizer; see note below.</p>

    <div class="gradcam-images">
      <div class="gradcam-panel">
        <div class="gradcam-panel-title">ORIGINAL IMAGE</div>
        <img src="${a.previewUrl}" alt="original" />
      </div>
      <div class="gradcam-panel">
        <div class="gradcam-panel-title">${CAM_METHOD_LABELS[gradcamMethod].toUpperCase()} <span id="gradcam-mode-label">OVERLAY</span></div>
        <div style="position:relative;">
          <img id="gradcam-base-img" src="${a.previewUrl}" alt="base" style="display:block;" />
          <img id="gradcam-heat-img" src="data:image/png;base64,${cam.heatmap}" alt="heatmap"
               style="position:absolute; inset:0; width:100%; height:100%; object-fit:cover; opacity:${gradcamIntensity / 100};" />
        </div>
      </div>
    </div>

    <div class="gradcam-controls">
      <div class="mode-toggle" id="mode-toggle">
        <button class="mode-btn" data-mode="original">Original</button>
        <button class="mode-btn" data-mode="heatmap">Heatmap</button>
        <button class="mode-btn" data-mode="overlay">Overlay</button>
      </div>
      <div class="intensity-row" id="intensity-row">
        <span style="font-size:0.8rem; color:var(--muted);">Overlay Intensity</span>
        <input type="range" min="0" max="100" value="${gradcamIntensity}" id="intensity-slider" />
        <span id="intensity-value" style="font-size:0.85rem; font-weight:bold; min-width:36px;">${gradcamIntensity}%</span>
      </div>
    </div>

    <div class="explain-box">
      <div class="explain-box-title">WHY DID THE MODEL MAKE THIS PREDICTION?</div>
      <p>The Grad-CAM visualization highlights image regions that received stronger activation during the model's prediction. These regions indicate where the model focused its attention when producing the classification.</p>
      <p style="margin:0;"><strong>Grad-CAM shows model attention, not definitive proof of manipulation or authenticity.</strong></p>
    </div>

    <div class="action-row">
      <button class="btn" id="download-viz-btn">&#8681; Download Visualization</button>
      <button class="btn btn-primary" id="gradcam-report-btn">&#128196; Generate Forensic Report</button>
      <button class="btn" id="gradcam-back-btn">&#8592; Back to Analysis</button>
    </div>
  `;

  document.querySelectorAll("#cam-method-toggle .mode-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.method === gradcamMethod);
    btn.addEventListener("click", () => {
      if (btn.dataset.method === gradcamMethod) return;
      switchCamMethod(btn.dataset.method, body);
    });
  });

  applyGradcamMode(gradcamMode);
  document.querySelectorAll("#mode-toggle .mode-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === gradcamMode);
    btn.addEventListener("click", () => { gradcamMode = btn.dataset.mode; applyGradcamMode(gradcamMode); });
  });
  document.getElementById("intensity-slider").addEventListener("input", (e) => {
    gradcamIntensity = Number(e.target.value);
    document.getElementById("intensity-value").textContent = `${gradcamIntensity}%`;
    if (gradcamMode === "overlay") document.getElementById("gradcam-heat-img").style.opacity = gradcamIntensity / 100;
  });
  document.getElementById("download-viz-btn").addEventListener("click", () => downloadGradcamVisualization());
  document.getElementById("gradcam-report-btn").addEventListener("click", () => goToPage("report"));
  document.getElementById("gradcam-back-btn").addEventListener("click", () => goToPage(activeAnalysis.pageKey));
}

function applyGradcamMode(mode) {
  document.querySelectorAll("#mode-toggle .mode-btn").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  const label = document.getElementById("gradcam-mode-label");
  const base = document.getElementById("gradcam-base-img");
  const heat = document.getElementById("gradcam-heat-img");
  const intensityRow = document.getElementById("intensity-row");
  if (!base || !heat) return;
  const cam = currentCamResult(activeAnalysis);

  if (mode === "original") {
    label.textContent = "ORIGINAL";
    base.src = activeAnalysis.previewUrl;
    heat.style.opacity = 0;
    intensityRow.style.visibility = "hidden";
  } else if (mode === "heatmap") {
    label.textContent = "HEATMAP";
    base.src = `data:image/png;base64,${cam.heatmap}`;
    heat.style.opacity = 0;
    intensityRow.style.visibility = "hidden";
  } else {
    label.textContent = "OVERLAY";
    base.src = activeAnalysis.previewUrl;
    heat.style.opacity = gradcamIntensity / 100;
    intensityRow.style.visibility = "visible";
  }
}

function downloadGradcamVisualization() {
  const cam = currentCamResult(activeAnalysis);
  const link = document.createElement("a");
  link.href = gradcamMode === "heatmap" ? `data:image/png;base64,${cam.heatmap}` : `data:image/png;base64,${cam.overlay}`;
  link.download = `${activeAnalysis.analysisId}_${gradcamMethod}_${gradcamMode}.png`;
  link.click();
}

/* ===================== Report page ===================== */
const INFERENCE_STEPS_APPLIED = ["Face Detection", "Image Resize", "Normalization", "Tensor Conversion"];
const TRAINING_AUGMENTATIONS = ["Random Resized Crop", "Random Horizontal Flip", "Color Jitter", "Channel Shift", "Gaussian Blur", "JPEG Compression", "Gaussian Noise"];

async function renderReportPage() {
  const body = document.getElementById("report-body");
  if (!activeAnalysis) {
    body.innerHTML = emptyStateHtml(
      "NO FORENSIC REPORT AVAILABLE",
      "Complete an image analysis before generating a report."
    );
    return;
  }
  const a = activeAnalysis;

  if (!a.overlayB64) {
    body.innerHTML = `
      <div class="card analyzing-card">
        <div class="spinner-ring"></div>
        <div class="analyzing-title">PREPARING REPORT</div>
        <div class="analyzing-sub">Computing the Grad-CAM explainability panel…</div>
      </div>
    `;
    try {
      const formData = new FormData();
      formData.append("file", await resolveUploadBlob(a), a.filename);
      const res = await fetch(`/gradcam?model=${a.modelKey}`, { method: "POST", body: formData });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      a.heatmapB64 = data.gradcam_heatmap;
      a.overlayB64 = data.gradcam_overlay;
    } catch (e) {
      body.innerHTML = `<p style="color:var(--fake); padding:16px;">Report generation failed: ${e.message}</p>`;
      return;
    }
  }
  const isReal = a.label === "Real";
  const verdictText = isReal ? "REAL / AUTHENTIC" : "AI GENERATED";
  const confidence = isReal ? a.realPct : a.fakePct;

  const preprocessRows = INFERENCE_STEPS_APPLIED.map((s) => `<tr><td>${s}</td><td class="tag-applied">&#10003; Applied</td></tr>`).join("");
  const augRows = TRAINING_AUGMENTATIONS.map((s) => `<li>${s}</li>`).join("");

  body.innerHTML = `
    <div class="report-doc">
      <div class="report-topbar">
        <div>
          <h1>AI IMAGE AUTHENTICITY<br/>FORENSIC ANALYSIS REPORT</h1>
        </div>
        <div class="download-dropdown">
          <button class="btn btn-primary" id="download-report-btn">&#128196; Download Report &#9662;</button>
          <div class="download-menu" id="download-menu">
            <button id="download-pdf-btn">&#128196; PDF</button>
            <button id="download-docx-btn">&#128196; Word Document (.docx)</button>
          </div>
        </div>
      </div>
      <div class="report-meta">Analysis ID: ${a.analysisId} &nbsp;|&nbsp; Generated: ${a.generatedAt}</div>

      <div class="report-grid-info">
        <div class="report-section">
          <div class="report-section-title"><span class="report-section-num">1</span> CASE INFORMATION</div>
          <div class="report-kv"><span class="k">Analysis ID</span><span>${a.analysisId}</span></div>
          <div class="report-kv"><span class="k">Date &amp; Time</span><span>${a.generatedAt}</span></div>
          <div class="report-kv"><span class="k">Uploaded Filename</span><span>${a.filename}</span></div>
        </div>
        <div class="report-section">
          <div class="report-section-title"><span class="report-section-num">2</span> IMAGE INFORMATION</div>
          <div class="report-kv"><span class="k">File Type</span><span>${a.fileType}</span></div>
          <div class="report-kv"><span class="k">Resolution</span><span>${a.resolution}</span></div>
          <div class="report-kv"><span class="k">File Size</span><span>${a.fileSize}</span></div>
          <div class="report-kv"><span class="k">Color Mode</span><span>RGB</span></div>
        </div>
        <div class="report-section">
          <div class="report-section-title"><span class="report-section-num">3</span> MODEL INFORMATION</div>
          <div class="report-kv"><span class="k">Model</span><span>${a.modelLabel}</span></div>
          <div class="report-kv"><span class="k">Model Version</span><span>${a.modelVersion}</span></div>
          <div class="report-kv"><span class="k">Prediction</span><span style="color:${isReal ? "var(--real)" : "var(--fake)"}; font-weight:bold;">${verdictText}</span></div>
          <div class="report-kv"><span class="k">Confidence</span><span>${confidence.toFixed(2)}%</span></div>
          <div class="report-kv"><span class="k">AI Probability</span><span>${a.fakePct.toFixed(2)}%</span></div>
          <div class="report-kv"><span class="k">Real Probability</span><span>${a.realPct.toFixed(2)}%</span></div>
        </div>
      </div>

      <div class="report-section">
        <div class="report-section-title"><span class="report-section-num">4</span> INFERENCE PREPROCESSING PIPELINE</div>
        <table class="preprocess-table"><tbody>${preprocessRows}</tbody></table>
        <div style="margin-top:14px; font-size:0.78rem; color:var(--muted); font-weight:bold;">TRAINING AUGMENTATION (used during training, NOT applied during inference)</div>
        <ul style="font-size:0.82rem; color:var(--muted); margin:8px 0 0 0;">${augRows}</ul>
      </div>

      <div class="report-section">
        <div class="report-section-title"><span class="report-section-num">5</span> EXPLAINABILITY ANALYSIS (GRAD-CAM)</div>
        <div class="report-images">
          <figure style="margin:0;"><img src="${a.previewUrl}" alt="original" /><figcaption>Original Image</figcaption></figure>
          <figure style="margin:0;"><img src="data:image/png;base64,${a.overlayB64}" alt="overlay" /><figcaption>Grad-CAM Overlay</figcaption></figure>
        </div>
        <p style="font-size:0.82rem; color:var(--muted); margin-top:12px; margin-bottom:0;">The Grad-CAM visualization highlights regions that received stronger activation during the model's prediction — this shows model attention, not definitive proof of manipulation or authenticity.</p>
      </div>

      ${a.meta ? `
      <div class="report-section">
        <div class="report-section-title"><span class="report-section-num">6</span> FORENSIC SCAN (META DETECTOR)</div>
        ${(() => {
          const m = a.meta;
          const md = (m.signals && m.signals.metadata) || {};
          const wm = (m.signals && m.signals.watermark) || {};
          const flags = [];
          if (wm.found) flags.push(`Invisible watermark: ${wm.label || wm.message}${wm.method ? " (" + wm.method + ")" : ""}`);
          if (md.c2pa && md.c2pa.length) flags.push(`C2PA / Content Credentials (${md.c2pa.length})`);
          if (md.synthetic_source_tag) flags.push("digitalSourceType marks image as synthetic (AI)");
          if (md.camera_markers && md.camera_markers.length) flags.push(`Camera metadata: ${md.camera_markers[0]}`);
          const evidence = (m.evidence && m.evidence.length) ? m.evidence : ["No strong forensic signals detected."];
          return `
            <div class="report-kv"><span class="k">Meta Verdict</span><span>${m.verdict}</span></div>
            <div class="report-kv"><span class="k">AI Score</span><span>${m.ai_score != null ? (m.ai_score * 100).toFixed(1) + "%" : "—"}</span></div>
            <div class="report-kv"><span class="k">Confidence</span><span>${m.confidence != null ? (m.confidence * 100).toFixed(1) + "%" : "—"}</span></div>
            ${flags.length ? `<div class="report-kv"><span class="k">Signals</span><span>${flags.join("; ")}</span></div>` : ""}
            <ul style="font-size:0.82rem; color:var(--muted); margin:10px 0 0 0; padding-left:20px;">
              ${evidence.map((e) => `<li style="margin-bottom:4px;">${e}</li>`).join("")}
            </ul>
            <p style="font-size:0.78rem; color:var(--muted); margin-top:12px; margin-bottom:0;">The meta-detector analyzes metadata, invisible watermarks, spectral content, sensor-noise statistics and double-JPEG artifacts — evidence independent of the neural model's learned features.</p>
          `;
        })()}
      </div>
      ` : ""}

      <div class="report-section">
        <div class="report-section-title"><span class="report-section-num">7</span> FINAL ASSESSMENT</div>
        <div class="final-assessment">
          <div class="verdict-icon ${isReal ? "" : ""}" style="color:${isReal ? "var(--real)" : "var(--fake)"};">${isReal ? "&#9989;" : "&#9888;&#65039;"}</div>
          <div>
            <div class="verdict-label" style="color:${isReal ? "var(--real)" : "var(--fake)"};">${verdictText}</div>
            <div class="verdict-conf" style="color:${isReal ? "var(--real)" : "var(--fake)"};">${confidence.toFixed(2)}%</div>
            <div class="verdict-conf-label">Confidence Score</div>
          </div>
          <p style="font-size:0.85rem; color:var(--muted); margin:0;">The model classified the submitted image as ${verdictText.toLowerCase()} with a confidence score of ${confidence.toFixed(2)}%. Based on the analysis, the likelihood of this image being artificially generated is ${a.fakePct >= 70 ? "high" : a.fakePct >= 40 ? "moderate" : "low"}.</p>
        </div>
      </div>

      <div class="report-section">
        <div class="report-section-title"><span class="report-section-num">8</span> DISCLAIMER</div>
        <div class="disclaimer-box">This report represents the output of an AI-based image authenticity detection model and should be interpreted as an analytical assessment rather than definitive proof of image manipulation or authenticity. It does not constitute a certified, legally admissible forensic conclusion.</div>
      </div>
    </div>
  `;

  const dropdownBtn = document.getElementById("download-report-btn");
  const menu = document.getElementById("download-menu");
  dropdownBtn.addEventListener("click", () => menu.classList.toggle("open"));
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".download-dropdown")) menu.classList.remove("open");
  }, { once: true });

  document.getElementById("download-pdf-btn").addEventListener("click", () => {
    menu.classList.remove("open");
    printReportAsPdf();
  });
  document.getElementById("download-docx-btn").addEventListener("click", () => {
    menu.classList.remove("open");
    downloadDocxReport();
  });
}

function printReportAsPdf() {
  const page = document.getElementById("page-report");
  page.classList.add("printing");
  window.print();
  window.addEventListener("afterprint", () => page.classList.remove("printing"), { once: true });
}

async function downloadDocxReport() {
  const a = activeAnalysis;
  const isReal = a.label === "Real";
  const payload = {
    analysis_id: a.analysisId,
    generated_at: a.generatedAt,
    filename: a.filename,
    file_type: a.fileType,
    resolution: a.resolution,
    file_size: a.fileSize,
    color_mode: "RGB",
    model_label: a.modelLabel,
    model_version: a.modelVersion,
    label: a.label,
    real_pct: a.realPct,
    fake_pct: a.fakePct,
    input_image_b64: a.previewUrl.startsWith("data:") ? a.previewUrl.split(",")[1] : await urlToB64(a.previewUrl),
    overlay_b64: a.overlayB64,
  };
  try {
    const res = await fetch("/report/docx", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${a.analysisId}_report.docx`;
    link.click();
    URL.revokeObjectURL(url);
  } catch (e) {
    alert(`Failed to generate Word report: ${e.message}`);
  }
}

async function urlToB64(url) {
  const res = await fetch(url);
  const blob = await res.blob();
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result.split(",")[1]);
    reader.readAsDataURL(blob);
  });
}

/* ===================== Analysis History page ===================== */
function historyStatusBadge(status) {
  if (status === "failed") return `<span class="badge" style="background:var(--fake-bg); color:var(--fake);">&#9888; Failed</span>`;
  return `<span class="badge" style="background:var(--real-bg); color:var(--real);">&#10003; Completed</span>`;
}

function getFilteredHistory() {
  const searchId = (document.getElementById("history-search-id")?.value || "").toLowerCase().trim();
  const searchFile = (document.getElementById("history-search-filename")?.value || "").toLowerCase().trim();
  const modelFilter = document.getElementById("history-filter-model")?.value || "all";
  const predFilter = document.getElementById("history-filter-prediction")?.value || "all";

  return analysisHistory.filter((r) => {
    if (searchId && !r.analysisId.toLowerCase().includes(searchId)) return false;
    if (searchFile && !r.filename.toLowerCase().includes(searchFile)) return false;
    if (modelFilter !== "all" && r.modelKey !== modelFilter) return false;
    if (predFilter !== "all") {
      if (predFilter === "failed" && r.status !== "failed") return false;
      if (predFilter === "real" && r.label !== "Real") return false;
      if (predFilter === "fake" && r.label !== "Fake") return false;
    }
    return true;
  });
}

function historyTableRowsHtml(rows) {
  if (!rows.length) {
    return `<tr><td colspan="7" style="text-align:center; padding:40px; color:var(--muted);">No analyses match these filters.</td></tr>`;
  }
  return rows.map((r) => {
    const isFailed = r.status === "failed";
    const isReal = r.label === "Real";
    const predictionCell = isFailed
      ? `<span style="color:var(--muted);">&mdash;</span>`
      : `<span style="color:${isReal ? "var(--real)" : "var(--fake)"}; font-weight:bold;">${isReal ? "AUTHENTIC" : "AI GENERATED"}</span>`;
    const confidenceCell = isFailed ? "&mdash;" : `${(isReal ? r.realPct : r.fakePct).toFixed(2)}%`;
    const actionCell = isFailed
      ? `<span style="color:var(--muted); font-size:0.78rem;">${r.errorMessage || "Analysis failed"}</span>`
      : `<button class="btn" style="padding:6px 14px; font-size:0.78rem;" onclick="viewHistoryEntry('${r.analysisId}')">View Details &#8594;</button>`;

    return `
      <tr style="border-bottom:1px solid var(--border);">
        <td style="padding:10px 8px; font-weight:bold;">${r.analysisId}</td>
        <td style="padding:10px 8px; color:var(--muted); font-size:0.82rem;">${r.generatedAt}</td>
        <td style="padding:10px 8px;">
          <div style="display:flex; align-items:center; gap:8px;">
            <img src="${r.previewUrl}" style="width:36px; height:36px; object-fit:cover; border-radius:6px; border:1px solid var(--border);" />
            <div>
              <div style="font-size:0.82rem;">${r.filename}</div>
              ${r.resolution ? `<div style="font-size:0.72rem; color:var(--muted);">${r.resolution}</div>` : ""}
            </div>
          </div>
        </td>
        <td style="padding:10px 8px; font-size:0.82rem;">${r.modelLabel}</td>
        <td style="padding:10px 8px;">${predictionCell}</td>
        <td style="padding:10px 8px;">${confidenceCell}</td>
        <td style="padding:10px 8px;">${historyStatusBadge(r.status)}</td>
        <td style="padding:10px 8px;">${actionCell}</td>
      </tr>
    `;
  }).join("");
}

function refreshHistoryTable() {
  const rows = getFilteredHistory();
  document.getElementById("history-table-body").innerHTML = historyTableRowsHtml(rows);
}

function viewHistoryEntry(analysisId) {
  const record = analysisHistory.find((r) => r.analysisId === analysisId);
  if (!record || record.status === "failed") return;
  activeAnalysis = record;
  goToPage("report");
}

let historyTimelineLimit = 5;

function groupHistoryByDate(records) {
  const groups = [];
  const byLabel = {};
  records.forEach((r) => {
    const dateLabel = (r.generatedAt || "").split(",").slice(0, 2).join(",").trim() || "Unknown date";
    if (!byLabel[dateLabel]) {
      byLabel[dateLabel] = { dateLabel, entries: [] };
      groups.push(byLabel[dateLabel]);
    }
    byLabel[dateLabel].entries.push(r);
  });
  return groups;
}

function timelinePredictionBadge(r) {
  if (r.status === "failed") return `<span class="badge" style="background:var(--fake-bg); color:var(--fake); font-size:0.68rem;">ANALYSIS FAILED</span>`;
  const isReal = r.label === "Real";
  return `<span class="badge" style="background:${isReal ? "var(--real-bg)" : "var(--fake-bg)"}; color:${isReal ? "var(--real)" : "var(--fake)"}; font-size:0.68rem;">${isReal ? "AUTHENTIC" : "AI GENERATED"}</span>`;
}

function historyTimelineHtml() {
  if (!analysisHistory.length) {
    return `<p style="color:var(--muted); font-size:0.85rem;">No analyses yet.</p>`;
  }
  const visible = analysisHistory.slice(0, historyTimelineLimit);
  const groups = groupHistoryByDate(visible);
  const groupsHtml = groups.map((g) => `
    <div class="timeline-date-header">${g.dateLabel}</div>
    ${g.entries.map((r) => {
      const time = (r.generatedAt || "").split(",").pop().trim();
      const confidence = r.status === "failed" ? "" : `<div class="timeline-confidence">${(r.label === "Real" ? r.realPct : r.fakePct).toFixed(2)}%</div>`;
      const action = r.status === "failed"
        ? `<span style="color:var(--muted); font-size:0.75rem;">${r.errorMessage || "Failed"}</span>`
        : `<a href="#" class="timeline-view-link" onclick="viewHistoryEntry('${r.analysisId}'); return false;">View Analysis &#8594;</a>`;
      return `
        <div class="timeline-entry">
          <span class="timeline-dot"></span>
          <div class="timeline-entry-body">
            <div class="timeline-time">${time}</div>
            <div class="timeline-id">${r.analysisId}</div>
            <div class="timeline-filename">${r.filename}</div>
            <div class="timeline-model">${r.modelLabel || ""}</div>
            ${timelinePredictionBadge(r)}
            ${confidence}
            <div>${action}</div>
          </div>
        </div>
      `;
    }).join("")}
  `).join("");

  const loadMoreHtml = historyTimelineLimit < analysisHistory.length
    ? `<button class="btn btn-block" id="timeline-load-more" style="margin-top:12px;">Load More &#8595;</button>`
    : "";

  return groupsHtml + loadMoreHtml;
}

function refreshTimeline() {
  document.getElementById("history-timeline-body").innerHTML = historyTimelineHtml();
  const loadMoreBtn = document.getElementById("timeline-load-more");
  if (loadMoreBtn) {
    loadMoreBtn.addEventListener("click", () => {
      historyTimelineLimit += 5;
      refreshTimeline();
    });
  }
}

function renderHistoryPage() {
  historyTimelineLimit = 5;
  const body = document.getElementById("history-body");
  const total = analysisHistory.length;
  const mainCount = analysisHistory.filter((r) => r.modelKey === "noaug").length;
  const crossCount = analysisHistory.filter((r) => r.modelKey === "cross_domain").length;
  const completedCount = analysisHistory.filter((r) => r.status === "completed").length;
  const successRate = total ? ((completedCount / total) * 100).toFixed(1) : "0.0";

  if (total === 0) {
    body.innerHTML = `
      <div class="page-header">
        <h1>ANALYSIS HISTORY</h1>
        <p class="page-desc">Review previous image authenticity analyses, predictions, model results, and forensic reports.</p>
      </div>
      <div class="empty-state">
        <div class="empty-state-icon">&#8986;</div>
        <div class="empty-state-title">NO ANALYSES YET</div>
        <div class="empty-state-sub">Run an analysis on the Main Model or Cross-Domain Model page and it will appear here.</div>
        <button class="btn btn-primary" onclick="goToPage('main')">Go to Main Model</button>
      </div>
    `;
    return;
  }

  body.innerHTML = `
    <div class="page-header">
      <h1>ANALYSIS HISTORY</h1>
      <p class="page-desc">Review previous image authenticity analyses, predictions, model results, and forensic reports.</p>
    </div>

    <div class="history-stats">
      <div class="history-stat-card">
        <div class="history-stat-label">TOTAL ANALYSES</div>
        <div class="history-stat-value">${total}</div>
      </div>
      <div class="history-stat-card">
        <div class="history-stat-label">MAIN MODEL</div>
        <div class="history-stat-value">${mainCount}</div>
      </div>
      <div class="history-stat-card">
        <div class="history-stat-label">CROSS-DOMAIN MODEL</div>
        <div class="history-stat-value">${crossCount}</div>
      </div>
      <div class="history-stat-card">
        <div class="history-stat-label">SUCCESS RATE</div>
        <div class="history-stat-value">${successRate}%</div>
      </div>
    </div>

    <div class="history-filters">
      <input type="text" id="history-search-id" class="history-input" placeholder="Search Analysis ID…" />
      <input type="text" id="history-search-filename" class="history-input" placeholder="Search Filename…" />
      <select id="history-filter-model" class="history-input">
        <option value="all">All Models</option>
        <option value="noaug">Main Model</option>
        <option value="cross_domain">Cross-Domain Model</option>
      </select>
      <select id="history-filter-prediction" class="history-input">
        <option value="all">All Results</option>
        <option value="real">Authentic</option>
        <option value="fake">AI Generated</option>
        <option value="failed">Failed</option>
      </select>
      <button class="btn" id="history-clear-filters">&#8635; Clear Filters</button>
    </div>

    <div class="history-layout">
      <div style="overflow-x:auto;">
        <table style="width:100%; border-collapse:collapse; font-size:0.85rem;">
          <thead>
            <tr style="border-bottom:2px solid var(--border); text-align:left; color:var(--muted); font-size:0.72rem; letter-spacing:0.04em;">
              <th style="padding:8px;">ANALYSIS ID</th>
              <th style="padding:8px;">DATE &amp; TIME</th>
              <th style="padding:8px;">IMAGE</th>
              <th style="padding:8px;">MODEL</th>
              <th style="padding:8px;">PREDICTION</th>
              <th style="padding:8px;">CONFIDENCE</th>
              <th style="padding:8px;">STATUS</th>
              <th style="padding:8px;">ACTION</th>
            </tr>
          </thead>
          <tbody id="history-table-body">${historyTableRowsHtml(analysisHistory)}</tbody>
        </table>
      </div>

      <div class="history-timeline-panel">
        <div class="history-timeline-title">&#8986; TIMELINE (Latest First)</div>
        <div id="history-timeline-body">${historyTimelineHtml()}</div>
      </div>
    </div>
  `;

  ["history-search-id", "history-search-filename"].forEach((id) => {
    document.getElementById(id).addEventListener("input", refreshHistoryTable);
  });
  ["history-filter-model", "history-filter-prediction"].forEach((id) => {
    document.getElementById(id).addEventListener("change", refreshHistoryTable);
  });
  document.getElementById("history-clear-filters").addEventListener("click", () => {
    document.getElementById("history-search-id").value = "";
    document.getElementById("history-search-filename").value = "";
    document.getElementById("history-filter-model").value = "all";
    document.getElementById("history-filter-prediction").value = "all";
    refreshHistoryTable();
  });

  const loadMoreBtn = document.getElementById("timeline-load-more");
  if (loadMoreBtn) {
    loadMoreBtn.addEventListener("click", () => {
      historyTimelineLimit += 5;
      refreshTimeline();
    });
  }
}

/* ===================== User Guide page ===================== */
/* ===================== Real-World Probe (benchmark) page ===================== */
function renderBenchmarkPage() {
  const body = document.getElementById("benchmark-body");
  body.innerHTML = `
    <div class="page-header">
      <h1>REAL-WORLD PROBE RESULTS</h1>
      <p class="page-desc">An independent re-test of the M5 "Real-Latest" cross-domain finding (originally 8.6% accuracy on 70 smartphone photos), extended with a matching AI-generated set. Neither original dataset is preserved in the repo, so both sides use 50-image proxies: real, recent Unsplash photography, and AI-generated faces (Stable Diffusion + StyleGAN2, bitmind/SyntheticFacesHQ).</p>
    </div>

    <h3 style="margin:22px 0 10px 0; font-size:1rem;">Real Photos (ground truth: Real, N=50)</h3>
    <div class="history-stats">
      <div class="history-stat-card">
        <div class="history-stat-label">BEST.PTH ACCURACY</div>
        <div class="history-stat-value">42.0%</div>
      </div>
      <div class="history-stat-card">
        <div class="history-stat-label">NOAUG ACCURACY (DEPLOYED)</div>
        <div class="history-stat-value">24.0%</div>
      </div>
      <div class="history-stat-card">
        <div class="history-stat-label">META-DETECTOR CORRECT</div>
        <div class="history-stat-value">70.0%</div>
      </div>
      <div class="history-stat-card">
        <div class="history-stat-label">META-DETECTOR WRONG</div>
        <div class="history-stat-value">0.0%</div>
      </div>
    </div>

    <h3 style="margin:22px 0 10px 0; font-size:1rem;">AI-Generated Photos (ground truth: Fake, N=50)</h3>
    <div class="history-stats">
      <div class="history-stat-card">
        <div class="history-stat-label">BEST.PTH ACCURACY</div>
        <div class="history-stat-value">44.0%</div>
      </div>
      <div class="history-stat-card">
        <div class="history-stat-label">NOAUG ACCURACY (DEPLOYED)</div>
        <div class="history-stat-value">26.0%</div>
      </div>
      <div class="history-stat-card">
        <div class="history-stat-label">META-DETECTOR CORRECT</div>
        <div class="history-stat-value">2.0%</div>
      </div>
      <div class="history-stat-card">
        <div class="history-stat-label">META-DETECTOR WRONG</div>
        <div class="history-stat-value">42.0%</div>
      </div>
    </div>

    <div class="guide-card" style="margin:20px 0;">
      <h4>CNN Model Comparison - Both Test Sets</h4>
      <p>Both deployed checkpoints run through the same face-crop -&gt; normalize -&gt; forward-pass pipeline used by <code>/predict</code>.</p>
      <table style="width:100%; border-collapse:collapse; font-size:0.85rem; margin-top:10px;">
        <thead>
          <tr style="border-bottom:1px solid var(--border); text-align:left;">
            <th style="padding:8px 6px;">Checkpoint</th>
            <th style="padding:8px 6px;">Real Photos (N=50)</th>
            <th style="padding:8px 6px;">AI-Generated (N=50)</th>
          </tr>
        </thead>
        <tbody>
          <tr style="border-bottom:1px solid var(--border);">
            <td style="padding:8px 6px;"><code>mobilenetv3_best.pth</code></td>
            <td style="padding:8px 6px;">42.0% (21/50 correct, 29 called Fake)</td>
            <td style="padding:8px 6px;">44.0% (22/50 correct, 28 called Real)</td>
          </tr>
          <tr>
            <td style="padding:8px 6px;"><code>mobilenetv3_noaug.pth</code> (production default)</td>
            <td style="padding:8px 6px;">24.0% (12/50 correct, 38 called Fake)</td>
            <td style="padding:8px 6px;">26.0% (13/50 correct, 37 called Real)</td>
          </tr>
        </tbody>
      </table>
      <div class="guide-note">Both checkpoints hover near chance on <em>both</em> test sets - not a one-sided "too trigger-happy" bias, just generally unreliable outside their training distribution. <code>noaug</code> (what's actually deployed) is worse than <code>best</code> on both sets, despite being chosen for its higher fake-recall on the held-out test set.</div>
    </div>

    <div class="guide-card" style="margin-bottom:20px;">
      <h4>Meta-Detector (Forensic Scan) - Both Test Sets</h4>
      <p>The separate EXIF/watermark/spectral/noise forensic scan (<code>meta_detector.py</code>, independent of the CNN), run on the same 100 images.</p>
      <table style="width:100%; border-collapse:collapse; font-size:0.85rem; margin-top:10px;">
        <thead>
          <tr style="border-bottom:1px solid var(--border); text-align:left;">
            <th style="padding:8px 6px;">Verdict</th>
            <th style="padding:8px 6px;">Real Photos (N=50)</th>
            <th style="padding:8px 6px;">AI-Generated (N=50)</th>
          </tr>
        </thead>
        <tbody>
          <tr style="border-bottom:1px solid var(--border);">
            <td style="padding:8px 6px;"><span class="badge badge-meta-real">Real / Likely real</span></td>
            <td style="padding:8px 6px; font-weight:bold;">35 (70.0%) - correct</td>
            <td style="padding:8px 6px; color:var(--fake);">21 (42.0%) - wrong</td>
          </tr>
          <tr style="border-bottom:1px solid var(--border);">
            <td style="padding:8px 6px;"><span class="badge badge-meta">Uncertain (mixed evidence)</span></td>
            <td style="padding:8px 6px;">15 (30.0%)</td>
            <td style="padding:8px 6px;">28 (56.0%)</td>
          </tr>
          <tr>
            <td style="padding:8px 6px;"><span class="badge badge-meta-fake">Likely/AI generated</span></td>
            <td style="padding:8px 6px; color:var(--fake);">0 (0.0%) - would be wrong</td>
            <td style="padding:8px 6px; font-weight:bold;">1 (2.0%) - correct</td>
          </tr>
        </tbody>
      </table>
      <div class="guide-note"><strong>Sharply asymmetric.</strong> On real photos the meta-detector is excellent at not crying wolf - zero confident false positives, worst case "uncertain." But on AI-generated images it's nearly blind - only 1 of 50 correctly flagged, and it never once reached the strongest "AI generated / AI edited" tier. This SyntheticFacesHQ set is curated/high-quality output with no embedded watermarks or suspicious EXIF, so there's little for a metadata-based scan to catch - it's tuned for authenticity-verification, not generation-detection, on this kind of clean synthetic output.</div>
    </div>

    <div class="guide-card">
      <h4>What does this mean for the 8.6% figure?</h4>
      <p>M5's original "Real-Latest" probe (Raunak, 70 smartphone photos, <code>mobilenetv3_best.pth</code>) found <strong>8.6% accuracy</strong> - 6 of 70 correct, a 91.4% false-positive rate. That exact dataset was never committed to the repo, so it can't be re-run precisely. This page's real-photo proxy (professional Unsplash photography rather than raw phone snapshots - a different but related distribution shift) reproduces the same underlying failure: <strong>42.0%</strong> on <code>best</code> and <strong>24.0%</strong> on <code>noaug</code>, both far below in-distribution accuracy (99.63% / 95.98%).</p>
      <p>There is no preserved pre-CelebA-HD baseline for this specific metric (M5 §4.1: the Stage 1+2 checkpoint was overwritten before that evaluation could run), so it's not possible to state precisely how much CelebA-HD changed real-world performance - only that, post-fix, it remains poor on both real and AI-generated out-of-distribution content. Neither signal alone is reliable here: the CNN is near-chance on both classes, and the meta-detector's strength (never falsely flagging real photos) is offset by near-blindness to this particular kind of clean synthetic output. A production system would need both signals combined, and ideally a wider net of AI-generation fingerprints than EXIF/watermark scanning alone catches.</p>
    </div>
  `;
}

function renderGuidePage() {
  const body = document.getElementById("guide-body");
  body.innerHTML = `
    <div class="guide-intro-grid">
      <div class="guide-intro-card">
        <div class="icon">&#128187;</div>
        <h3>What is this system?</h3>
        <p>Face Forensics AI Authenticity is an AI-powered system that analyzes face images to determine whether they are Real (Authentic) or AI Generated.</p>
      </div>
      <div class="guide-intro-card">
        <div class="icon">&#9881;&#65039;</div>
        <h3>Main Model</h3>
        <p>The Main Model is our primary face authenticity detection model, trained to provide high-accuracy classification.</p>
      </div>
      <div class="guide-intro-card">
        <div class="icon">&#127760;</div>
        <h3>Cross-Domain Model</h3>
        <p>The Cross-Domain Model is designed to generalize across different image domains, cameras, and generation sources.</p>
      </div>
      <div class="guide-intro-card">
        <div class="icon">&#128161;</div>
        <h3>Why Two Models?</h3>
        <p>Using two complementary models helps improve reliability and reduces false predictions across diverse real-world scenarios.</p>
      </div>
    </div>

    <div class="guide-steps">
      ${["Select Model", "Upload Image", "Click Analyze", "Review Result", "Open Grad-CAM", "Generate Report"]
        .map((label, i, arr) => `
          <div class="guide-step">
            <div class="circle">${i + 1}</div>
            <div class="label">${label}</div>
          </div>
          ${i < arr.length - 1 ? '<div class="guide-arrow">&#8594;</div>' : ""}
        `).join("")}
    </div>

    <div class="guide-grid-2">
      <div class="guide-card">
        <h4>Understanding Confidence</h4>
        <p>The confidence score (0%–100%) indicates how strongly the model believes in its prediction.</p>
        <ul>
          <li>90% and above: Very High Confidence</li>
          <li>70%–90%: High Confidence</li>
          <li>50%–70%: Moderate Confidence</li>
          <li>Below 50%: Low Confidence</li>
        </ul>
        <div class="guide-note">Confidence reflects model certainty, not absolute truth.</div>
      </div>
      <div class="guide-card">
        <h4>Using Grad-CAM</h4>
        <p>Grad-CAM highlights the image regions that had the strongest influence on the model's prediction.</p>
        <ul>
          <li>Red/Yellow areas = High activation</li>
          <li>Blue areas = Low activation</li>
          <li>Overlay intensity slider adjusts blend strength</li>
        </ul>
        <div class="guide-note">Grad-CAM shows model attention, not definitive proof of manipulation.</div>
      </div>
      <div class="guide-card">
        <h4>Understanding the Report</h4>
        <p>The forensic report includes all important information about the analysis:</p>
        <ul>
          <li>Case &amp; Image Information</li>
          <li>Model Information &amp; Result</li>
          <li>Preprocessing Steps (Inference)</li>
          <li>Grad-CAM Visualization</li>
          <li>Final Assessment &amp; Disclaimer</li>
        </ul>
        <div class="guide-note">Reports can be downloaded as PDF or Word documents.</div>
      </div>
      <div class="guide-card">
        <h4>Important Disclaimer</h4>
        <ul>
          <li>Results should be interpreted as an analytical assessment only.</li>
          <li>This is not a substitute for professional forensic investigation.</li>
          <li>We do not guarantee 100% accuracy in all scenarios.</li>
        </ul>
        <div class="guide-note guide-warning">The system does not provide absolute proof of authenticity or manipulation.</div>
      </div>
    </div>

    <div class="guide-grid-2">
      <div class="guide-card">
        <h4>No Analysis Available (Grad-CAM)</h4>
        <p>If you open Grad-CAM without completing an analysis, you will see an empty state prompting you to run an analysis first.</p>
        <button class="btn" onclick="goToPage('main')">Go to Main Model</button>
      </div>
      <div class="guide-card">
        <h4>No Report Available</h4>
        <p>If you open Forensic Report without completing an analysis, you will see an empty state prompting you to run an analysis first.</p>
        <button class="btn" onclick="goToPage('main')">Go to Main Model</button>
      </div>
    </div>
  `;
}

/* ===================== Init ===================== */
initTheme();
initNav();
initMainModelToggle();
initModelDetailsModal();
loadPersistedHistory();
checkHealth();
