const healthLine = document.getElementById("health-line");
const globalError = document.getElementById("global-error");

let availableModels = [];

function showGlobalError(message) {
  globalError.textContent = message;
  globalError.classList.remove("hidden");
}

const CLASSES_OK = true; // no-op, keeps linter quiet about unused pattern

const MODEL_LABELS = {
  best: "Best (CelebA / 3-stage)",
  noaug: "No-Augmentation",
  manipulations: "Manipulation-Robust",
  cross_domain: "Cross-Domain",
  tuned: "Tuned Hyperparameters",
};

const FACE_ALIGNMENT_LABELS = {
  retinaface: "RetinaFace",
  center_crop_fallback: "Center-crop (RetinaFace unavailable on this machine)",
};

// ---------- Health check ----------
async function checkHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    availableModels = data.loaded_models || [];
    const parts = [];
    parts.push(
      availableModels.length
        ? `Models in use: ${availableModels.map((m) => MODEL_LABELS[m] || m).join(", ")}`
        : "No models loaded yet"
    );
    parts.push(`Face alignment: ${FACE_ALIGNMENT_LABELS[data.face_alignment] || data.face_alignment}`);
    healthLine.textContent = parts.join(" · ");

    if (!availableModels.includes("best")) {
      showGlobalError(
        "No 'best' checkpoint is loaded on the server. Place mobilenetv3_best1.pth " +
        "in the checkpoints folder (or point CHECKPOINT_DIR at it) and restart the server."
      );
      disableAllAnalyzeButtons(true);
    }
  } catch (e) {
    healthLine.textContent = "Backend unreachable.";
    showGlobalError("Could not reach the backend at /health. Is uvicorn running?");
    disableAllAnalyzeButtons(true);
  }
}

function disableAllAnalyzeButtons(disabled) {
  document.querySelectorAll(".analyze-btn").forEach((btn) => {
    btn.disabled = disabled;
  });
}

// ---------- Generic upload-widget wiring ----------
// Wires drag/drop + click + file-input + preview + Analyze button for one
// section, identified by `prefix` (matches the -predict/-crossdomain/
// -manipulation/-compare id suffixes in index.html). Returns nothing; the
// caller supplies onAnalyze(file) to run when the button is clicked.
function setupUploadWidget(prefix, onAnalyze) {
  const dropZone = document.getElementById(`drop-zone-${prefix}`);
  const fileInput = document.getElementById(`file-input-${prefix}`);
  const previewWrap = document.getElementById(`preview-wrap-${prefix}`);
  const previewImg = document.getElementById(`preview-img-${prefix}`);
  const analyzeBtn = document.getElementById(`analyze-btn-${prefix}`);
  const errorEl = document.getElementById(`error-${prefix}`);

  let selectedFile = null;

  function showError(message) {
    errorEl.textContent = message;
    errorEl.classList.remove("hidden");
  }
  function clearError() {
    errorEl.classList.add("hidden");
    errorEl.textContent = "";
  }

  function setFile(file) {
    if (!file.type.startsWith("image/")) {
      showError("Please choose an image file.");
      return;
    }
    clearError();
    selectedFile = file;
    previewImg.src = URL.createObjectURL(file);
    previewWrap.classList.remove("hidden");
    analyzeBtn.disabled = availableModels.length === 0;
  }

  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) setFile(fileInput.files[0]);
  });

  analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    clearError();
    analyzeBtn.disabled = true;
    try {
      await onAnalyze(selectedFile, previewImg.src);
    } catch (e) {
      showError(`Request failed: ${e}`);
    } finally {
      analyzeBtn.disabled = false;
    }
  });
}

function setBody(id, html) {
  document.getElementById(id).innerHTML = html;
}

function badgeHtml(label, realPct, fakePct) {
  const cls = label === "Real" ? "real" : "fake";
  const pct = label === "Real" ? realPct : fakePct;
  return `
    <span class="badge ${cls}">${label}</span>
    <div style="margin-top:8px;">Real: ${realPct.toFixed(2)}% &nbsp;|&nbsp; Fake: ${fakePct.toFixed(2)}%</div>
    <div class="confidence-bar"><div style="width:${pct.toFixed(1)}%"></div></div>
  `;
}

// ---------- Shared: run /predict against a given model + render badge/Grad-CAM ----------
async function runPredictAndRender(bodyId, modelKey, file, previewSrc, showReportButton = false) {
  setBody(bodyId, `<span class="spinner">Running prediction + Grad-CAM (${modelKey})…</span>`);

  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`/predict?model=${modelKey}`, { method: "POST", body: formData });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `HTTP ${res.status}`);
  }
  const data = await res.json();
  const p = data.prediction;

  const reportBtnHtml = showReportButton
    ? `<button type="button" class="link-btn" id="report-btn-${bodyId}" style="margin-top:10px;">Generate Forensic Report</button>
       <span id="report-status-${bodyId}" class="placeholder" style="margin-left:8px;"></span>`
    : "";

  setBody(
    bodyId,
    badgeHtml(p.label, p.real_pct, p.fake_pct) +
      `<div class="placeholder" style="margin:8px 0;">Model: ${modelKey} · Face alignment used: ${data.face_alignment_used}</div>
      <div class="gradcam-grid">
        <figure><img src="${previewSrc}" /><figcaption>Input</figcaption></figure>
        <figure><img src="data:image/png;base64,${data.gradcam_heatmap}" /><figcaption>Heatmap (${p.label})</figcaption></figure>
        <figure><img src="data:image/png;base64,${data.gradcam_overlay}" /><figcaption>Overlay</figcaption></figure>
      </div>
      ${reportBtnHtml}`
  );

  if (showReportButton) {
    document.getElementById(`report-btn-${bodyId}`).addEventListener("click", () => generateForensicReport(file, modelKey, bodyId));
  }
}

// ---------- Forensic report generation (opens a standalone printable HTML report) ----------
async function generateForensicReport(file, modelKey, bodyId) {
  const statusEl = document.getElementById(`report-status-${bodyId}`);
  statusEl.textContent = "Generating report…";
  try {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(`/report?model=${modelKey}`, { method: "POST", body: formData });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const html = await res.text();
    const blob = new Blob([html], { type: "text/html" });
    const url = URL.createObjectURL(blob);
    window.open(url, "_blank");
    statusEl.textContent = "Report opened in a new tab.";
  } catch (e) {
    statusEl.textContent = `Failed to generate report: ${e}`;
  }
}

// ---------- 1. Prediction + Grad-CAM (main model, best1) ----------
setupUploadWidget("predict", (file, previewSrc) => runPredictAndRender("predict-body", "best", file, previewSrc, true));

// ---------- 2. No-Augmentation Model (independent upload, highest accuracy) ----------
setupUploadWidget("noaug", (file, previewSrc) => runPredictAndRender("noaug-body", "noaug", file, previewSrc));

// ---------- 2. Cross-Domain Testing (dedicated cross_domain model) ----------
setupUploadWidget("crossdomain", async (file, previewSrc) => {
  setBody("crossdomain-body", `<span class="spinner">Running prediction…</span>`);

  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch("/predict?model=cross_domain", { method: "POST", body: formData });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `HTTP ${res.status}`);
  }
  const data = await res.json();
  const p = data.prediction;

  setBody(
    "crossdomain-body",
    `<div style="display:flex; gap:16px; align-items:flex-start; flex-wrap:wrap;">
      <img src="${previewSrc}" style="width:140px; border-radius:8px; border:1px solid var(--border);" />
      <div>${badgeHtml(p.label, p.real_pct, p.fake_pct)}</div>
    </div>
    <p class="placeholder" style="margin-top:10px;">
      Prediction from the dedicated cross-domain model (trained on general,
      non-face images across multiple domains), not the main face model.
    </p>`
  );
});

// ---------- 3. Manipulation Robustness ----------
// Just a verdict + confidence score - nothing else. Verdict = majority
// vote across all 11 manipulation predictions; confidence = average
// confidence of the manipulations that agree with that verdict.
setupUploadWidget("manipulation", async (file) => {
  setBody("manipulation-body", `<span class="spinner">Running 11 manipulations…</span>`);

  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch("/robustness", { method: "POST", body: formData });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  const rows = data.rows;

  const fakeRows = rows.filter((r) => r.label === "Fake");
  const realRows = rows.filter((r) => r.label === "Real");
  const verdict = fakeRows.length >= realRows.length ? "Fake" : "Real";
  const supporting = verdict === "Fake" ? fakeRows : realRows;
  const avgConfidence =
    supporting.reduce((sum, r) => sum + (verdict === "Fake" ? r.fake_pct : r.real_pct), 0) / supporting.length;

  const rowsHtml = rows
    .map(
      (r) => `
        <tr>
          <td><img class="thumb" src="data:image/png;base64,${r.thumbnail}" /></td>
          <td>${r.manipulation}</td>
          <td><span class="badge ${r.label === "Real" ? "real" : "fake"}">${r.label}</span></td>
          <td>${r.real_pct.toFixed(1)}%</td>
          <td>${r.fake_pct.toFixed(1)}%</td>
        </tr>`
    )
    .join("");

  setBody(
    "manipulation-body",
    `<span class="badge ${verdict === "Fake" ? "fake" : "real"}" style="font-size:1.05rem; padding:6px 16px;">${verdict}</span>
    <div style="margin-top:8px;">Confidence: ${avgConfidence.toFixed(1)}%</div>
    <div class="confidence-bar"><div style="width:${avgConfidence.toFixed(1)}%"></div></div>
    <button type="button" class="link-btn" id="manip-details-toggle">Show detailed breakdown (for report)</button>
    <div id="manip-details" class="hidden" style="margin-top:12px;">
      <table>
        <thead><tr><th></th><th>Manipulation</th><th>Prediction</th><th>Real%</th><th>Fake%</th></tr></thead>
        <tbody>${rowsHtml}</tbody>
      </table>
    </div>`
  );

  document.getElementById("manip-details-toggle").addEventListener("click", (e) => {
    const details = document.getElementById("manip-details");
    const isHidden = details.classList.toggle("hidden");
    e.target.textContent = isHidden ? "Show detailed breakdown (for report)" : "Hide detailed breakdown";
  });
});

// ---------- 4b. Test All 3 Models on One Image ----------
const THREE_MODELS = ["best", "noaug", "manipulations"];

async function runThreeModels(file) {
  const runOne = async (modelKey) => {
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch(`/predict?model=${modelKey}`, { method: "POST", body: formData });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        return { modelKey, error: detail.detail || `HTTP ${res.status}` };
      }
      const data = await res.json();
      return { modelKey, prediction: data.prediction };
    } catch (e) {
      return { modelKey, error: String(e) };
    }
  };
  return Promise.all(THREE_MODELS.map(runOne));
}

function threeModelsTableHtml(outcomes) {
  const rowsHtml = outcomes
    .map((o) => {
      if (o.error) {
        return `<tr><td><code>${o.modelKey}</code></td><td colspan="4" class="placeholder">${o.error}</td></tr>`;
      }
      const p = o.prediction;
      const pct = p.label === "Real" ? p.real_pct : p.fake_pct;
      return `
        <tr>
          <td><code>${o.modelKey}</code></td>
          <td><span class="badge ${p.label === "Real" ? "real" : "fake"}">${p.label}</span></td>
          <td>${p.real_pct.toFixed(2)}%</td>
          <td>${p.fake_pct.toFixed(2)}%</td>
          <td><div class="confidence-bar" style="width:100px;"><div style="width:${pct.toFixed(1)}%"></div></div></td>
        </tr>`;
    })
    .join("");
  return `<div style="overflow-x:auto;">
    <table>
      <thead><tr><th>Model</th><th>Prediction</th><th>Real%</th><th>Fake%</th><th>Confidence</th></tr></thead>
      <tbody>${rowsHtml}</tbody>
    </table>
  </div>`;
}

setupUploadWidget("compare3", async (file, previewSrc) => {
  setBody("compare3-body", `<span class="spinner">Running all 3 models…</span>`);
  const outcomes = await runThreeModels(file);
  setBody(
    "compare3-body",
    `<div style="display:flex; gap:16px; align-items:flex-start; flex-wrap:wrap; margin-bottom:12px;">
      <img src="${previewSrc}" style="width:140px; border-radius:8px; border:1px solid var(--border);" />
    </div>
    ${threeModelsTableHtml(outcomes)}`
  );
});

// ---------- 4a. Pre-loaded example: one real + one fake, all 3 models ----------
async function runSampleCompare() {
  const samples = [
    { url: "sample_real.jpg", label: "Real (ground truth)" },
    { url: "sample_fake.png", label: "Fake (ground truth)" },
  ];

  try {
    const blocks = await Promise.all(
      samples.map(async (s) => {
        const res = await fetch(s.url);
        const blob = await res.blob();
        const file = new File([blob], s.url, { type: blob.type });
        const outcomes = await runThreeModels(file);
        return `<div style="flex:1; min-width:260px;">
          <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
            <img src="${s.url}" style="width:100px; border-radius:8px; border:1px solid var(--border);" />
            <strong>${s.label}</strong>
          </div>
          ${threeModelsTableHtml(outcomes)}
        </div>`;
      })
    );
    setBody("sample-compare-body", `<div style="display:flex; gap:24px; flex-wrap:wrap;">${blocks.join("")}</div>`);
  } catch (e) {
    setBody("sample-compare-body", `<span class="error">Failed to run sample comparison: ${e}</span>`);
  }
}

// ---------- 3a. Pre-loaded example: one real + one fake, cross_domain model ----------
async function runCrossDomainExample() {
  const samples = [
    { url: "sample_cross_real.jpg", label: "Real (ground truth)" },
    { url: "sample_cross_fake.png", label: "Fake (ground truth)" },
  ];

  try {
    const blocks = await Promise.all(
      samples.map(async (s) => {
        const res = await fetch(s.url);
        const blob = await res.blob();
        const file = new File([blob], s.url, { type: blob.type });
        const formData = new FormData();
        formData.append("file", file);
        const predRes = await fetch("/predict?model=cross_domain", { method: "POST", body: formData });
        if (!predRes.ok) throw new Error(`HTTP ${predRes.status}`);
        const data = await predRes.json();
        const p = data.prediction;
        return `<div style="flex:1; min-width:220px; display:flex; align-items:center; gap:14px;">
          <img src="${s.url}" style="width:90px; height:90px; object-fit:cover; border-radius:8px; border:1px solid var(--border);" />
          <div>
            <div class="placeholder" style="margin-bottom:4px;">${s.label}</div>
            ${badgeHtml(p.label, p.real_pct, p.fake_pct)}
          </div>
        </div>`;
      })
    );
    setBody("cross-example-body", `<div style="display:flex; gap:24px; flex-wrap:wrap;">${blocks.join("")}</div>`);
  } catch (e) {
    setBody("cross-example-body", `<span class="error">Failed to run cross-domain example: ${e}</span>`);
  }
}

// ---------- 4c. Pairwise Model Comparison ----------
setupUploadWidget("compare", async (file) => {
  document.querySelectorAll("#compare-card .compare-body").forEach(
    (el) => (el.innerHTML = `<span class="spinner">Comparing models…</span>`)
  );

  await Promise.all([
    runCompare(file, "augmentation", "compare-augmentation"),
    runCompare(file, "hparams", "compare-hparams"),
  ]);
});

async function runCompare(file, mode, containerId) {
  const formData = new FormData();
  formData.append("file", file);
  const el = document.querySelector(`#${containerId} .compare-body`);
  try {
    const res = await fetch(`/compare?mode=${mode}`, { method: "POST", body: formData });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (!data.available) {
      el.innerHTML = `<span class="placeholder">${data.reason || "Not available."}</span>`;
      return;
    }
    const names = Object.keys(data.results);
    el.innerHTML = names
      .map((name) => {
        const r = data.results[name];
        return `<div class="compare-model"><div class="name">${name}</div>${badgeHtml(r.label, r.real_pct, r.fake_pct)}</div>`;
      })
      .join("");
  } catch (e) {
    el.innerHTML = `<span class="error">Comparison failed: ${e}</span>`;
  }
}

// ---------- 5. Training & Evaluation Results (pre-computed, not live) ----------
function imgFig(url, caption) {
  return `<figure><img src="${url}" loading="lazy" /><figcaption>${caption}</figcaption></figure>`;
}

function renderGenericTable(rows) {
  if (!rows || rows.length === 0) return `<p class="placeholder">No data.</p>`;
  const headers = Object.keys(rows[0]);
  const head = headers.map((h) => `<th>${h}</th>`).join("");
  const body = rows
    .map((row) => `<tr>${headers.map((h) => `<td>${row[h]}</td>`).join("")}</tr>`)
    .join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function renderGradcamGroup(title, urls) {
  if (!urls || !urls.length) return "";
  const figs = urls.map((url) => imgFig(url, title)).join("");
  return `<div class="results-section"><h3>Grad-CAM — ${title} Predictions</h3><div class="image-gallery gradcam-gallery">${figs}</div></div>`;
}

const TABLE_TITLES = {
  robustness: "Robustness (corruption accuracy)",
  manipulation: "Manipulation Testing (Priority 3)",
  augmentation_ablation: "Augmentation Ablation (Priority 1)",
  cross_domain: "Cross-Domain Testing (Priority 4)",
  domain_accuracy_cross: "Cross-Domain Model — Per-Domain Validation Accuracy",
};

async function loadTrainingResults() {
  const container = document.getElementById("training-results-body");
  try {
    const res = await fetch("/api/training-results");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const images = data.images || {};

    let html = "";

    // Row 1: Confusion Matrix + Preprocessing Samples
    const row1 = [];
    if (images.confusion_matrix) row1.push(imgFig(images.confusion_matrix, "Confusion Matrix"));
    if (images.preprocessing_samples) row1.push(imgFig(images.preprocessing_samples, "Preprocessing Samples"));
    if (row1.length) {
      html += `<div class="results-section"><h3>Confusion Matrix &amp; Preprocessing</h3><div class="image-gallery gallery-2col">${row1.join("")}</div></div>`;
    }

    // Row 2: Augmentation Samples + Manipulation Samples
    const row2 = [];
    if (images.augmentation_samples) row2.push(imgFig(images.augmentation_samples, "Augmentation Samples"));
    if (images.manipulation_samples) row2.push(imgFig(images.manipulation_samples, "Manipulation Samples"));
    if (row2.length) {
      html += `<div class="results-section"><h3>Augmentation &amp; Manipulation Samples</h3><div class="image-gallery gallery-2col">${row2.join("")}</div></div>`;
    }

    // Row 3: Cross-Domain Samples, on its own
    if (images.cross_domain_samples) {
      html += `<div class="results-section"><h3>Cross-Domain Samples</h3><div class="image-gallery gallery-2col">${imgFig(images.cross_domain_samples, "Cross-Domain Samples")}</div></div>`;
    }

    // Row 4: Cross-Domain Model's own confusion matrix + training curves
    const row4 = [];
    if (images.confusion_matrix_cross_domain) row4.push(imgFig(images.confusion_matrix_cross_domain, "Cross-Domain Model — Confusion Matrix"));
    if (images.training_curves_cross) row4.push(imgFig(images.training_curves_cross, "Cross-Domain Model — Training Curves"));
    if (row4.length) {
      html += `<div class="results-section"><h3>Cross-Domain Model — Training Results</h3><div class="image-gallery gallery-2col">${row4.join("")}</div></div>`;
    }

    // Grad-CAM correct/incorrect galleries, capped at 3 per row
    const gallery = data.gradcam_gallery || {};
    html += renderGradcamGroup("Correct", gallery.correct);
    html += renderGradcamGroup("Incorrect", gallery.incorrect);

    // Tables
    const tableKeys = Object.keys(data.tables || {});
    if (tableKeys.length) {
      const blocks = tableKeys
        .map((key) => `<div><h3>${TABLE_TITLES[key] || key}</h3>${renderGenericTable(data.tables[key])}</div>`)
        .join("");
      html += `<div class="results-section"><div class="results-tables">${blocks}</div></div>`;
    }

    if (!html) {
      html = `<p class="placeholder">No result artifacts found in Deepfake/output/.</p>`;
    }

    container.innerHTML = html;
  } catch (e) {
    container.innerHTML = `<span class="error">Failed to load training results: ${e}</span>`;
  }
}

checkHealth();
runCrossDomainExample();
loadTrainingResults();
runSampleCompare();
