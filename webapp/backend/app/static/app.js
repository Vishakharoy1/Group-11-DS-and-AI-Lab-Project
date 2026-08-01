const healthLine = document.getElementById("health-line");
const globalError = document.getElementById("global-error");

let availableModels = [];

function showGlobalError(message) {
  globalError.textContent = message;
  globalError.classList.remove("hidden");
}

const CLASSES_OK = true; // no-op, keeps linter quiet about unused pattern

// ---------- Health check ----------
async function checkHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    availableModels = data.loaded_models || [];
    const parts = [];
    parts.push(availableModels.length ? `Loaded models: ${availableModels.join(", ")}` : "No models loaded yet");
    parts.push(`Face alignment: ${data.face_alignment}`);
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

// ---------- 1. Prediction + Grad-CAM ----------
setupUploadWidget("predict", async (file, previewSrc) => {
  setBody("predict-body", `<span class="spinner">Running prediction + Grad-CAM…</span>`);

  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch("/predict", { method: "POST", body: formData });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  const p = data.prediction;

  setBody(
    "predict-body",
    badgeHtml(p.label, p.real_pct, p.fake_pct) +
      `<div class="placeholder" style="margin:8px 0;">Face alignment used: ${data.face_alignment_used}</div>
      <div class="gradcam-grid">
        <figure><img src="${previewSrc}" /><figcaption>Input</figcaption></figure>
        <figure><img src="data:image/png;base64,${data.gradcam_heatmap}" /><figcaption>Heatmap (${p.label})</figcaption></figure>
        <figure><img src="data:image/png;base64,${data.gradcam_overlay}" /><figcaption>Overlay</figcaption></figure>
      </div>`
  );
});

// ---------- 2. Cross-Domain Testing ----------
setupUploadWidget("crossdomain", async (file, previewSrc) => {
  setBody("crossdomain-body", `<span class="spinner">Running prediction…</span>`);

  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch("/predict", { method: "POST", body: formData });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  const p = data.prediction;

  setBody(
    "crossdomain-body",
    `<div style="display:flex; gap:16px; align-items:flex-start; flex-wrap:wrap;">
      <img src="${previewSrc}" style="width:140px; border-radius:8px; border:1px solid var(--border);" />
      <div>${badgeHtml(p.label, p.real_pct, p.fake_pct)}</div>
    </div>
    <p class="placeholder" style="margin-top:10px;">
      See Section 5's Cross-Domain Testing table for the measured face_main vs.
      nano_banana accuracy gap — treat this prediction with reduced confidence
      if your image is far outside the face domain.
    </p>`
  );
});

// ---------- 3. Manipulation Robustness ----------
setupUploadWidget("manipulation", async (file) => {
  setBody("manipulation-body", `<span class="spinner">Running 11 manipulations…</span>`);

  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch("/robustness", { method: "POST", body: formData });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();

  const originalRow = data.rows.find((r) => r.manipulation === "original");
  const baseline = originalRow ? originalRow.label : null;

  const rowsHtml = data.rows
    .map((r) => {
      const flipped = baseline && r.label !== baseline;
      return `
        <tr class="${flipped ? "flipped" : ""}">
          <td><img class="thumb" src="data:image/png;base64,${r.thumbnail}" /></td>
          <td>${r.manipulation}</td>
          <td><span class="badge ${r.label === "Real" ? "real" : "fake"}">${r.label}</span></td>
          <td>${r.real_pct.toFixed(1)}%</td>
          <td>${r.fake_pct.toFixed(1)}%</td>
        </tr>`;
    })
    .join("");

  setBody(
    "manipulation-body",
    `<table>
      <thead><tr><th></th><th>Manipulation</th><th>Prediction</th><th>Real%</th><th>Fake%</th></tr></thead>
      <tbody>${rowsHtml}</tbody>
    </table>
    <p class="placeholder" style="margin-top:8px;">Rows highlighted red flipped away from the original prediction.</p>`
  );
});

// ---------- 4. Model Comparison ----------
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
loadTrainingResults();
