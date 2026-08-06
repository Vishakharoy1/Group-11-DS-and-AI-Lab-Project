const pptxgen = require("pptxgenjs");

// ---------------- Palette (Midnight Executive, forensic/security theme) ----------------
const NAVY = "1E2761";
const NAVY_DARK = "141B47";
const ICE = "CADCFC";
const ICE_SOFT = "EAF1FE";
const WHITE = "FFFFFF";
const GREEN = "2C5F2D";
const GREEN_SOFT = "E8F1E8";
const RED = "990011";
const RED_SOFT = "FBE9EA";
const GRAY = "5A5A66";
const GRAY_LIGHT = "F4F5F8";
const TEXT_DARK = "1A1A1A";

const IMG = {
  arch: "C:/Users/Vishakha.Roy/Downloads/Deepfake/Group-11-DS-and-AI-Lab-Project/images/mobilenetv3_pipeline_v3.png",
  cm: "C:/Users/Vishakha.Roy/Downloads/Deepfake/Group-11-DS-and-AI-Lab-Project/doc/Milestone-5/images/confusion_matrix_best_model.png",
  roc: "C:/Users/Vishakha.Roy/Downloads/Deepfake/Group-11-DS-and-AI-Lab-Project/doc/Milestone-5/images/local_roc_curve.png",
  pr: "C:/Users/Vishakha.Roy/Downloads/Deepfake/Group-11-DS-and-AI-Lab-Project/doc/Milestone-5/images/local_pr_curve.png",
  gcCorrect: "C:/Users/Vishakha.Roy/Downloads/Deepfake/Group-11-DS-and-AI-Lab-Project/doc/Milestone-5/images/somendu_gradcam_correct_fake.png",
  gcFailure: "C:/Users/Vishakha.Roy/Downloads/Deepfake/Group-11-DS-and-AI-Lab-Project/doc/Milestone-5/images/somendu_gradcam_failure_real.png",
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
const PW = 13.33, PH = 7.5;

// ---------------- Helpers ----------------
const TOTAL_SLIDES = 19;
function pageNum(slide, n) {
  slide.addText(String(n).padStart(2, "0") + " / " + TOTAL_SLIDES, {
    x: PW - 1.1, y: PH - 0.42, w: 0.9, h: 0.3, fontSize: 9, color: GRAY,
    align: "right", fontFace: "Calibri", margin: 0,
  });
}

function footerBrand(slide, dark) {
  slide.addText("MILESTONE 5  ·  FACE AUTHENTICITY DETECTION", {
    x: 0.55, y: PH - 0.42, w: 6, h: 0.3, fontSize: 8.5, color: dark ? "8C97C4" : "9AA0AE",
    fontFace: "Calibri", charSpacing: 1, margin: 0,
  });
}

function contentTitle(slide, kicker, title) {
  slide.addText(kicker.toUpperCase(), {
    x: 0.55, y: 0.42, w: 10, h: 0.3, fontSize: 12, bold: true, color: NAVY,
    fontFace: "Calibri", charSpacing: 1.2, margin: 0,
  });
  slide.addText(title, {
    x: 0.55, y: 0.72, w: 12.2, h: 0.75, fontSize: 30, bold: true, color: TEXT_DARK,
    fontFace: "Cambria", margin: 0,
  });
}

function badge(slide, x, y, d, num, color) {
  slide.addShape("ellipse", { x, y, w: d, h: d, fill: { color }, line: { type: "none" } });
  slide.addText(String(num), {
    x, y, w: d, h: d, align: "center", valign: "middle", fontSize: d > 0.5 ? 16 : 12,
    bold: true, color: WHITE, fontFace: "Calibri", margin: 0,
  });
}

function card(slide, x, y, w, h, fill) {
  slide.addShape("roundRect", {
    x, y, w, h, rectRadius: 0.08, fill: { color: fill || GRAY_LIGHT },
    line: { type: "none" }, shadow: { type: "outer", color: "000000", opacity: 0.12, blur: 6, offset: 2, angle: 90 },
  });
}

function statTile(slide, x, y, w, h, value, label, color) {
  card(slide, x, y, w, h, WHITE);
  slide.addText(value, {
    x, y: y + 0.12, w, h: h * 0.55, align: "center", valign: "bottom",
    fontSize: 30, bold: true, color, fontFace: "Cambria", margin: 0,
  });
  slide.addText(label, {
    x: x + 0.1, y: y + h * 0.62, w: w - 0.2, h: h * 0.34, align: "center", valign: "top",
    fontSize: 10.5, color: GRAY, fontFace: "Calibri", margin: 0,
  });
}

// ============================================================
// SLIDE 1 — TITLE
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addShape("ellipse", { x: 9.6, y: -2.2, w: 6.5, h: 6.5, fill: { color: NAVY_DARK }, line: { type: "none" } });
  s.addShape("ellipse", { x: -2.5, y: 4.5, w: 5, h: 5, fill: { color: NAVY_DARK }, line: { type: "none" } });

  s.addText("MILESTONE 5", {
    x: 0.9, y: 2.15, w: 8, h: 0.5, fontSize: 16, bold: true, color: ICE,
    fontFace: "Calibri", charSpacing: 3, margin: 0,
  });
  s.addText("Model Evaluation & Analysis", {
    x: 0.85, y: 2.6, w: 11.5, h: 1.5, fontSize: 46, bold: true, color: WHITE,
    fontFace: "Cambria", margin: 0,
  });
  s.addText("Deep Learning-Based Human Face Authenticity Detection", {
    x: 0.9, y: 3.85, w: 10.5, h: 0.6, fontSize: 18, color: ICE,
    fontFace: "Calibri", italic: true, margin: 0,
  });

  s.addShape("line", { x: 0.9, y: 4.7, w: 2.4, h: 0, line: { color: ICE, width: 1.5 } });

  s.addText("Group 11  ·  Data Science & AI Lab Project", {
    x: 0.9, y: 4.9, w: 8, h: 0.4, fontSize: 13, color: WHITE, fontFace: "Calibri", margin: 0,
  });
  s.addText("mobilenetv3_best.pth  —  final-mobilenet (1).ipynb", {
    x: 0.9, y: 6.65, w: 8, h: 0.3, fontSize: 10.5, color: "8C97C4", fontFace: "Courier New", margin: 0,
  });
}

// ============================================================
// SLIDE 2 — TEAM & CONTRIBUTIONS
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  contentTitle(s, "The Team", "Contributions — Milestone 5");

  const members = [
    ["Vishakha", "Pipeline & Presentation Lead", "Built the local web app (FastAPI + automated forensic report generation), diagnosed the preprocessing pipeline, and wrote Sections 1, 9 & 10.", GREEN],
    ["Rohit", "Training Stability", "Quantitative benchmarking — training configuration and real held-out test set results (Section 4).", NAVY],
    ["Aman", "Preprocessing & Transfer Learning", "Dataset setup and metric justification with real, verified numbers (Sections 2 & 3).", NAVY],
    ["Raunak", "Dataset & Bias Analysis", "Comprehensive error analysis — shortcut-learning root cause and cross-domain evaluation (Section 5).", RED],
    ["Somendu", "Explainability & Optimisation", "Grad-CAM verification, model limitations, and actionable insights (Sections 6, 7 & 8).", NAVY],
  ];

  let y = 1.75;
  const rowH = 1.02;
  members.forEach(([name, role, task, color], i) => {
    card(s, 0.55, y, 12.2, rowH - 0.14, GRAY_LIGHT);
    badge(s, 0.8, y + (rowH - 0.14) / 2 - 0.28, 0.56, name[0], color);
    s.addText(name, { x: 1.55, y: y + 0.08, w: 3.1, h: 0.4, fontSize: 15, bold: true, color: TEXT_DARK, fontFace: "Calibri", margin: 0 });
    s.addText(role, { x: 1.55, y: y + 0.44, w: 3.1, h: 0.35, fontSize: 10, italic: true, color, fontFace: "Calibri", margin: 0 });
    s.addText(task, { x: 4.8, y: y + 0.06, w: 7.75, h: rowH - 0.26, fontSize: 10.8, color: GRAY, fontFace: "Calibri", valign: "middle", margin: 0 });
    y += rowH;
  });

  footerBrand(s, false);
  pageNum(s, 2);
}

// ============================================================
// SLIDE 3 — PROJECT JOURNEY M1 -> M5
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  contentTitle(s, "Project Timeline", "The Journey — Milestone 1 to 5");

  const steps = [
    ["M1", "Proposed an explainable ViT + FFT/DCT frequency-fusion framework."],
    ["M2", "Built the dataset foundation — FFHQ vs. StyleGAN/SD, 120,000+ images."],
    ["M3", "Architecture bake-off: selected MobileNetV3-Large — 4.2M params, 99.96% in-domain, 80% OOD (ChatGPT/Gemini)."],
    ["M4", "Delivered a 2-stage checkpoint, 99.06% test accuracy — faculty review flagged shortcut learning + a preprocessing mismatch."],
    ["M5", "Root-caused the issue, added a Stage 3 CelebA-HD fix, built the forensic web app, and ran a fully honest re-evaluation."],
  ];

  const x0 = 0.9, xEnd = 12.4, yLine = 4.0;
  s.addShape("line", { x: x0, y: yLine, w: xEnd - x0, h: 0, line: { color: ICE, width: 3 } });

  const n = steps.length;
  const gap = (xEnd - x0) / (n - 1);
  steps.forEach(([tag, desc], i) => {
    const cx = x0 + gap * i;
    const isM5 = tag === "M5";
    badge(s, cx - 0.28, yLine - 0.28, 0.56, tag, isM5 ? RED : NAVY);
    const top = i % 2 === 0;
    const boxW = 2.55;
    const boxX = Math.min(Math.max(cx - boxW / 2, 0.4), PW - boxW - 0.4);
    const boxY = top ? yLine - 1.85 : yLine + 0.55;
    card(s, boxX, boxY, boxW, 1.25, isM5 ? RED_SOFT : ICE_SOFT);
    s.addText(desc, {
      x: boxX + 0.15, y: boxY + 0.1, w: boxW - 0.3, h: 1.05, fontSize: 9.6, color: TEXT_DARK,
      fontFace: "Calibri", valign: "middle", margin: 0,
    });
    const connY1 = top ? boxY + 1.25 : yLine;
    const connY2 = top ? yLine : boxY;
    s.addShape("line", { x: cx, y: Math.min(connY1, connY2), w: 0, h: Math.abs(connY2 - connY1), line: { color: "C9C9CF", width: 1, dashType: "dash" } });
  });

  footerBrand(s, false);
  pageNum(s, 3);
}

// ============================================================
// SLIDE 4 — M5 OBJECTIVES
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  contentTitle(s, "Scope", "Milestone 5 — Objectives");

  const objs = [
    "Final evaluation on a strictly held-out test set — Accuracy, Precision, Recall, F1, ROC-AUC",
    "Investigate and document the root cause of the shortcut-learning misclassification issue",
    "Verify preprocessing consistency between the training notebook and the deployed UI backend",
    "Quantify robustness under real-world image manipulations — tints, JPEG, blur, noise",
    "Verify explainability via Grad-CAM, confirming the model attends to facial regions",
    "Assess deployment readiness — latency, model size, and quantization potential",
    "Compile all findings into a viva-ready report and presentation",
  ];

  let y = 1.85;
  const rowH = 0.68;
  objs.forEach((t, i) => {
    badge(s, 0.7, y + 0.03, 0.4, i + 1, i === 1 ? RED : NAVY);
    s.addText(t, {
      x: 1.35, y: y - 0.05, w: 11.2, h: 0.6, fontSize: 13.5, color: TEXT_DARK,
      fontFace: "Calibri", valign: "middle", margin: 0,
    });
    if (i < objs.length - 1) {
      s.addShape("line", { x: 1.35, y: y + rowH - 0.08, w: 11.2, h: 0, line: { color: "EDEDF0", width: 1 } });
    }
    y += rowH;
  });

  footerBrand(s, false);
  pageNum(s, 4);
}

// ============================================================
// SLIDE 5 — DATASET & EVALUATION SETUP
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  contentTitle(s, "Section 2", "Dataset & Evaluation Setup");

  // Left: data sources
  s.addText("DATA SOURCES", { x: 0.55, y: 1.75, w: 5, h: 0.3, fontSize: 11, bold: true, color: NAVY, charSpacing: 1, fontFace: "Calibri", margin: 0 });
  const sources = [
    ["FFHQ (real)", "70,000 available", GREEN],
    ["Stable Diffusion (fake)", "9,001 available", RED],
    ["CelebA-HD (real, Stage 3 only)", "8,000 added", GREEN],
  ];
  let sy = 2.15;
  sources.forEach(([name, n, c]) => {
    card(s, 0.55, sy, 5.7, 0.72, GRAY_LIGHT);
    s.addShape("ellipse", { x: 0.75, y: sy + 0.24, w: 0.24, h: 0.24, fill: { color: c }, line: { type: "none" } });
    s.addText(name, { x: 1.15, y: sy + 0.08, w: 3.6, h: 0.3, fontSize: 12, bold: true, color: TEXT_DARK, fontFace: "Calibri", margin: 0 });
    s.addText(n, { x: 1.15, y: sy + 0.36, w: 3.6, h: 0.3, fontSize: 10, color: GRAY, fontFace: "Calibri", margin: 0 });
    sy += 0.86;
  });
  s.addText("Sampled to 15,000 Real + 9,001 Fake = 24,001 images total, then split 80:10:10 (stratified, SEED=42).", {
    x: 0.55, y: sy + 0.05, w: 5.7, h: 0.7, fontSize: 10.5, italic: true, color: GRAY, fontFace: "Calibri", margin: 0,
  });

  // Right: split table
  s.addText("TRAIN / VAL / TEST SPLIT", { x: 6.65, y: 1.75, w: 6, h: 0.3, fontSize: 11, bold: true, color: NAVY, charSpacing: 1, fontFace: "Calibri", margin: 0 });
  const rows = [
    [{ text: "Split", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
     { text: "Real", options: { bold: true, color: WHITE, fill: { color: NAVY }, align: "right" } },
     { text: "Fake", options: { bold: true, color: WHITE, fill: { color: NAVY }, align: "right" } },
     { text: "Total", options: { bold: true, color: WHITE, fill: { color: NAVY }, align: "right" } }],
    [{ text: "Training" }, { text: "12,000", options: { align: "right" } }, { text: "7,200", options: { align: "right" } }, { text: "19,200", options: { align: "right", bold: true } }],
    [{ text: "Validation" }, { text: "1,500", options: { align: "right" } }, { text: "900", options: { align: "right" } }, { text: "2,400", options: { align: "right", bold: true } }],
    [{ text: "Test (held-out)", options: { bold: true, fill: { color: ICE_SOFT } } }, { text: "1,500", options: { align: "right", fill: { color: ICE_SOFT } } }, { text: "901", options: { align: "right", fill: { color: ICE_SOFT } } }, { text: "2,401", options: { align: "right", bold: true, fill: { color: ICE_SOFT } } }],
  ];
  s.addTable(rows, {
    x: 6.65, y: 2.15, w: 6.1, h: 1.9, fontSize: 12, fontFace: "Calibri", color: TEXT_DARK,
    border: { type: "solid", color: "E5E5EA", pt: 1 }, autoPage: false,
  });
  s.addText("Real:Fake ratio (~62.5:37.5) preserved identically across all three splits.", {
    x: 6.65, y: 4.25, w: 6.1, h: 0.4, fontSize: 10, italic: true, color: GRAY, fontFace: "Calibri", margin: 0,
  });

  card(s, 6.65, 4.85, 6.1, 1.55, RED_SOFT);
  s.addText("Why CelebA-HD?", { x: 6.9, y: 5.0, w: 5.6, h: 0.3, fontSize: 12, bold: true, color: RED, fontFace: "Calibri", margin: 0 });
  s.addText("M4's faculty review flagged real, modern smartphone photos being misclassified as fake. 8,000 CelebA-HD real photos were added in Stage 3 specifically to correct this.", {
    x: 6.9, y: 5.35, w: 5.6, h: 0.95, fontSize: 10.5, color: TEXT_DARK, fontFace: "Calibri", margin: 0,
  });

  footerBrand(s, false);
  pageNum(s, 5);
}

// ============================================================
// SLIDE 6 — ARCHITECTURE DIAGRAM
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  contentTitle(s, "Section 1 · Section 4.1", "Model Architecture & Pipeline — MobileNetV3-Large");

  const archAspect = 3723 / 2108;
  const archH = 5.55;
  const archW = archH * archAspect;
  s.addImage({ path: IMG.arch, x: (PW - archW) / 2, y: 1.6, h: archH, w: archW });

  footerBrand(s, false);
  pageNum(s, 6);
}

// ============================================================
// SLIDE 7 — THREE-STAGE TRAINING STRATEGY
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  contentTitle(s, "Section 4.1", "Three-Stage Transfer Learning Strategy");

  const stages = [
    ["STAGE 1", "Classifier head only\nbackbone frozen", "3 epochs · LR 3e-4", "98.75%", ICE, NAVY],
    ["STAGE 2", "Unfreeze last 25%\n(blocks 12-16 of 17)", "7 epochs · LR 1e-5", "99.71%", "8FB3E8", WHITE],
    ["STAGE 3", "Full unfreeze\n+ CelebA-HD real photos", "3 epochs · LR 5e-6", "99.71%", NAVY, WHITE],
  ];
  const w = 3.7, gap = 0.5, x0 = (PW - (w * 3 + gap * 2)) / 2, y0 = 2.0, h = 3.2;
  stages.forEach(([tag, desc, hp, acc, fill, tc], i) => {
    const x = x0 + i * (w + gap);
    card(s, x, y0, w, h, fill);
    s.addText(tag, { x, y: y0 + 0.25, w, h: 0.4, align: "center", fontSize: 16, bold: true, color: tc, fontFace: "Calibri", margin: 0 });
    s.addText(desc, { x: x + 0.2, y: y0 + 0.8, w: w - 0.4, h: 0.85, align: "center", fontSize: 11.5, color: tc, fontFace: "Calibri", margin: 0 });
    s.addText(hp, { x, y: y0 + 1.65, w, h: 0.35, align: "center", italic: true, fontSize: 10.5, color: tc, fontFace: "Calibri", margin: 0 });
    s.addShape("line", { x: x + 0.6, y: y0 + 2.15, w: w - 1.2, h: 0, line: { color: tc, width: 0.75, transparency: 40 } });
    s.addText(acc, { x, y: y0 + 2.25, w, h: 0.6, align: "center", fontSize: 26, bold: true, color: tc, fontFace: "Cambria", margin: 0 });
    s.addText("best val. accuracy", { x, y: y0 + 2.85, w, h: 0.3, align: "center", fontSize: 9, color: tc, fontFace: "Calibri", margin: 0 });
    if (i < 2) {
      s.addShape("triangle", { x: x + w + 0.08, y: y0 + h / 2 - 0.13, w: 0.32, h: 0.26, rotate: 90, fill: { color: GRAY }, line: { type: "none" } });
    }
  });

  s.addText("Stage 3 targets the modern-smartphone-photo shortcut-learning failure identified in Milestone 5.", {
    x: 0.8, y: 5.55, w: 11.7, h: 0.4, align: "center", italic: true, fontSize: 11.5, color: GRAY, fontFace: "Calibri", margin: 0,
  });

  footerBrand(s, false);
  pageNum(s, 7);
}

// ============================================================
// SLIDE 8 — METRIC SELECTION JUSTIFICATION
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  contentTitle(s, "Section 3", "Metric Selection & Justification");

  const rows = [
    [
      { text: "Metric", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
      { text: "Purpose", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
      { text: "Result for this checkpoint", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
    ],
    [{ text: "Accuracy" }, { text: "Overall balance" }, { text: "99.63% (held-out)  vs.  8.6% (Real-Latest OOD)" }],
    [{ text: "Precision" }, { text: "Cost of false alarms" }, { text: "0.9993 Real (held-out)  vs.  91.4% false-positive rate (OOD)" }],
    [{ text: "Recall" }, { text: "Cost of missed fakes" }, { text: "0.9989 Fake — weak point is false alarms, not missed fakes" }],
    [{ text: "F1-score" }, { text: "Imbalanced-class trade-off" }, { text: "Real 0.9970 / Fake 0.9950 / Macro 0.9960" }],
    [{ text: "ROC-AUC" }, { text: "Threshold-independent ranking" }, { text: "Not computed on real test set (notebook gap) — 0.5856 on local sample" }],
    [{ text: "Cross-Domain Eval." }, { text: "Generalization to unseen distributions" }, { text: "100.0% (Real-Old)  vs.  8.6% (Real-Latest) — central finding" }],
  ];
  s.addTable(rows, {
    x: 0.55, y: 1.8, w: 12.2, h: 4.8, fontSize: 11.5, fontFace: "Calibri", color: TEXT_DARK,
    border: { type: "solid", color: "E5E5EA", pt: 1 }, autoPage: false,
    colW: [2.3, 3.4, 6.5],
  });

  footerBrand(s, false);
  pageNum(s, 8);
}

// ============================================================
// SLIDE 9 — QUANTITATIVE RESULTS
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  contentTitle(s, "Section 4.2", "Quantitative Performance — Held-Out Test Set");

  const stats = [
    ["99.63%", "Test Accuracy", NAVY],
    ["99.53%", "Macro Precision", NAVY],
    ["99.68%", "Macro Recall", NAVY],
    ["99.60%", "Macro F1-score", NAVY],
  ];
  const sw = 2.7, sh = 1.3, sx0 = 0.55, sy0 = 1.85, sgap = 0.15;
  stats.forEach(([v, l, c], i) => statTile(s, sx0 + i * (sw + sgap), sy0, sw, sh, v, l, c));

  s.addImage({ path: IMG.cm, x: 1.4, y: 3.5, h: 3.55, w: 3.55 * (586 / 590) });

  card(s, 5.5, 3.5, 6.7, 3.55, GRAY_LIGHT);
  s.addText("2,401-image test set — only 9 errors", { x: 5.75, y: 3.7, w: 6.2, h: 0.4, fontSize: 14, bold: true, color: TEXT_DARK, fontFace: "Calibri", margin: 0 });
  const bullets = [
    "8 Real images misclassified as Fake",
    "1 Fake image misclassified as Real",
    "Real: Precision 0.9993 · Recall 0.9947 · F1 0.9970",
    "Fake: Precision 0.9912 · Recall 0.9989 · F1 0.9950",
  ];
  s.addText(bullets.map((t, i) => ({ text: t, options: { bullet: { code: "2022" }, color: TEXT_DARK, fontSize: 12.5, breakLine: i < bullets.length - 1, paraSpaceAfter: 10 } })), {
    x: 5.75, y: 4.2, w: 6.2, h: 2.7, fontFace: "Calibri", valign: "top", margin: 0,
  });

  footerBrand(s, false);
  pageNum(s, 9);
}

// ============================================================
// SLIDE 10 — LIVE MULTI-MODEL TESTING (WEB APP)
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  contentTitle(s, "How Testing Is Done", "Live Multi-Model Testing — The Web App");

  s.addText(
    "Pre-loaded samples from Test_real_vs_Fake/ — no upload needed, runs automatically against all 3 loaded checkpoints (best, noaug, manipulations).",
    { x: 0.55, y: 1.6, w: 12.2, h: 0.4, fontSize: 11.5, italic: true, color: GRAY, fontFace: "Calibri", margin: 0 }
  );

  function predRow(model, label, realPct, fakePct) {
    const c = label === "Real" ? GREEN : RED;
    return [
      { text: model },
      { text: label, options: { bold: true, color: c } },
      { text: realPct, options: { align: "right" } },
      { text: fakePct, options: { align: "right" } },
    ];
  }

  const headerRow = [
    { text: "Model", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
    { text: "Prediction", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
    { text: "Real %", options: { bold: true, color: WHITE, fill: { color: NAVY }, align: "right" } },
    { text: "Fake %", options: { bold: true, color: WHITE, fill: { color: NAVY }, align: "right" } },
  ];

  const halfW = 5.75, gapW = 0.7, tableY = 2.5;
  s.addText("Real (ground truth)", { x: 0.55, y: tableY - 0.4, w: halfW, h: 0.35, fontSize: 13, bold: true, color: GREEN, fontFace: "Calibri", margin: 0 });
  s.addTable([
    headerRow,
    predRow("best", "Real", "91.11%", "8.89%"),
    predRow("noaug", "Fake", "19.46%", "80.54%"),
    predRow("manipulations", "Real", "99.99%", "0.01%"),
  ], { x: 0.55, y: tableY, w: halfW, h: 1.8, fontSize: 12, fontFace: "Calibri", color: TEXT_DARK, border: { type: "solid", color: "E5E5EA", pt: 1 }, autoPage: false });

  const rightX = 0.55 + halfW + gapW;
  s.addText("Fake (ground truth)", { x: rightX, y: tableY - 0.4, w: halfW, h: 0.35, fontSize: 13, bold: true, color: RED, fontFace: "Calibri", margin: 0 });
  s.addTable([
    headerRow,
    predRow("best", "Fake", "0.17%", "99.83%"),
    predRow("noaug", "Fake", "0.04%", "99.96%"),
    predRow("manipulations", "Fake", "0.41%", "99.59%"),
  ], { x: rightX, y: tableY, w: halfW, h: 1.8, fontSize: 12, fontFace: "Calibri", color: TEXT_DARK, border: { type: "solid", color: "E5E5EA", pt: 1 }, autoPage: false });

  card(s, 0.55, 4.85, 12.2, 1.85, GRAY_LIGHT);
  s.addText("What this live check shows", { x: 0.85, y: 5.02, w: 11.6, h: 0.35, fontSize: 13, bold: true, color: NAVY, fontFace: "Calibri", margin: 0 });
  const liveNotes = [
    "All 3 models agree correctly on the Fake sample — high confidence (99.6%+) across the board.",
    "On the Real sample, noaug is the only model that gets it wrong (misclassifies it as Fake, 80.54% confidence) — visible, direct evidence of the class-bias finding used to select best as the primary model over noaug elsewhere in this report.",
    "This tool runs live against the actual deployed checkpoints, not a static offline script — the same numbers anyone testing the web app today would see.",
  ];
  s.addText(
    liveNotes.map((t, i) => ({ text: t, options: { bullet: { code: "2022" }, color: TEXT_DARK, fontSize: 11, breakLine: i < liveNotes.length - 1, paraSpaceAfter: 8 } })),
    { x: 0.85, y: 5.4, w: 11.6, h: 1.25, fontFace: "Calibri", valign: "top", margin: 0 }
  );

  footerBrand(s, false);
  pageNum(s, 10);
}

// ============================================================
// SLIDE 11 — ROC CURVE
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  contentTitle(s, "Section 4.5", "ROC Curve");

  s.addImage({ path: IMG.roc, x: 0.9, y: 1.55, h: 5.35, w: 5.35 * (825 / 750) });

  card(s, 6.9, 1.85, 5.9, 2.0, ICE_SOFT);
  s.addText("AUC = 0.5856", { x: 7.15, y: 2.05, w: 5.4, h: 0.6, fontSize: 26, bold: true, color: NAVY, fontFace: "Cambria", margin: 0 });
  s.addText("Only slightly better than random guessing (0.5).", { x: 7.15, y: 2.65, w: 5.4, h: 1.0, fontSize: 12.5, color: TEXT_DARK, fontFace: "Calibri", margin: 0 });

  card(s, 6.9, 4.1, 5.9, 2.8, GRAY_LIGHT);
  s.addText("What this ROC-AUC indicates", { x: 7.15, y: 4.3, w: 5.4, h: 0.4, fontSize: 13, bold: true, color: NAVY, fontFace: "Calibri", margin: 0 });
  const rocPoints = [
    "0.5856 means the model can barely rank a Fake image above a Real image on this sample — its confidence scores carry little discriminative signal here.",
    "1.0 = perfect separation, 0.5 = random guessing — 0.5856 sits close to the random-guess line, visible in the curve tracking near the diagonal.",
    "Consistent with the 62.0% accuracy on this same sample (Section 4.5) — this is the same domain-shift effect, reflected in the ranking metric, not just the raw accuracy number.",
  ];
  s.addText(
    rocPoints.map((t, i) => ({ text: t, options: { bullet: { code: "2022" }, color: TEXT_DARK, fontSize: 11, breakLine: i < rocPoints.length - 1, paraSpaceAfter: 10 } })),
    { x: 7.15, y: 4.75, w: 5.4, h: 2.05, fontFace: "Calibri", valign: "top", margin: 0 }
  );

  footerBrand(s, false);
  pageNum(s, 11);
}

// ============================================================
// SLIDE 12 — PRECISION-RECALL CURVE
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  contentTitle(s, "Section 4.5", "Precision-Recall Curve");

  s.addImage({ path: IMG.pr, x: 0.9, y: 1.55, h: 5.35, w: 5.35 * (825 / 750) });

  card(s, 6.9, 1.85, 5.9, 2.0, ICE_SOFT);
  s.addText("AP = 0.6612", { x: 7.15, y: 2.05, w: 5.4, h: 0.6, fontSize: 26, bold: true, color: NAVY, fontFace: "Cambria", margin: 0 });
  s.addText("Average Precision on the same local supplementary sample.", { x: 7.15, y: 2.65, w: 5.4, h: 1.0, fontSize: 12.5, color: TEXT_DARK, fontFace: "Calibri", margin: 0 });

  card(s, 6.9, 4.1, 5.9, 2.8, GRAY_LIGHT);
  s.addText("Why this still matters", { x: 7.15, y: 4.3, w: 5.4, h: 0.4, fontSize: 13, bold: true, color: NAVY, fontFace: "Calibri", margin: 0 });
  s.addText(
    "62% accuracy on this same sample (Section 4.5) does not contradict the real 99.63% held-out result — it is independent confirmation of the exact domain-shift problem documented in Section 5, using a completely different image set.",
    { x: 7.15, y: 4.75, w: 5.4, h: 2.0, fontSize: 11.5, color: TEXT_DARK, fontFace: "Calibri", margin: 0 }
  );

  footerBrand(s, false);
  pageNum(s, 12);
}

// ============================================================
// SLIDE 13 — COMPREHENSIVE ERROR ANALYSIS
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  contentTitle(s, "Section 5", "Comprehensive Error Analysis");

  const cols = [
    ["Real-Old", "FFHQ-distribution images", "73", "100.0%", GREEN, GREEN_SOFT, "0.99997"],
    ["Real-Latest", "Recent smartphone photos (2025-26)", "70", "8.6%", RED, RED_SOFT, "0.00060"],
  ];
  const cw = 5.7, cx0 = 0.75, cgap = 0.5, cy = 1.65, ch = 3.55;
  cols.forEach(([tag, desc, n, acc, color, soft, medP], i) => {
    const x = cx0 + i * (cw + cgap);
    card(s, x, cy, cw, ch, soft);
    s.addText(tag, { x: x + 0.3, y: cy + 0.2, w: cw - 0.6, h: 0.35, fontSize: 15, bold: true, color, fontFace: "Calibri", margin: 0 });
    s.addText(desc, { x: x + 0.3, y: cy + 0.53, w: cw - 0.6, h: 0.35, fontSize: 10.5, italic: true, color: GRAY, fontFace: "Calibri", margin: 0 });
    s.addText(acc, { x: x + 0.3, y: cy + 0.9, w: cw - 0.6, h: 0.9, fontSize: 46, bold: true, color, fontFace: "Cambria", margin: 0 });
    s.addText("accuracy on " + n + " images", { x: x + 0.3, y: cy + 1.85, w: cw - 0.6, h: 0.3, fontSize: 10.5, color: GRAY, fontFace: "Calibri", margin: 0 });
    s.addShape("line", { x: x + 0.3, y: cy + 2.22, w: cw - 0.6, h: 0, line: { color, width: 0.75, transparency: 40 } });
    s.addText([
      { text: "Median P(Real):  ", options: { bold: true, color: TEXT_DARK } },
      { text: medP, options: { color } },
    ], { x: x + 0.3, y: cy + 2.35, w: cw - 0.6, h: 0.35, fontSize: 11.5, fontFace: "Calibri", margin: 0 });
    s.addText(
      i === 0 ? "The model correctly recognized every FFHQ-like image — resembles training data." :
                "Genuine, high-confidence misclassifications, not uncertain borderline calls.",
      { x: x + 0.3, y: cy + 2.7, w: cw - 0.6, h: 0.55, fontSize: 10, color: GRAY, fontFace: "Calibri", margin: 0 }
    );
  });

  card(s, 0.75, 5.35, 11.85, 1.62, GRAY_LIGHT);
  s.addText("In plain terms: what \"Real-Latest = 8.6%\" means", { x: 1.0, y: 5.47, w: 11.35, h: 0.3, fontSize: 12, bold: true, color: NAVY, fontFace: "Calibri", margin: 0 });
  s.addText(
    "Of 70 genuine, real, recent (2025–26) smartphone photographs tested, the model correctly identified only 6 of them as Real (8.6%) — it incorrectly flagged the other 64 as Fake (91.4%), despite every one of those 70 images actually being real people's genuine photos. Contrast with Real-Old (FFHQ-style images resembling training data), where it got 100% right. Same model, same checkpoint — only the photo's capture era/device differs. That gap is the central evidence for the shortcut-learning problem this milestone documents.",
    { x: 1.0, y: 5.78, w: 11.35, h: 1.15, fontSize: 10.5, color: TEXT_DARK, fontFace: "Calibri", margin: 0 }
  );

  footerBrand(s, false);
  pageNum(s, 13);
}

// ============================================================
// SLIDE 14 — ROOT CAUSE: SHORTCUT LEARNING
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  contentTitle(s, "Section 5.4", "Root Cause — Shortcut Learning");

  s.addText(
    "The model learned dataset-specific characteristics rather than intrinsic facial authenticity cues — a classic case of domain shift causing shortcut learning.",
    { x: 0.55, y: 1.65, w: 12.2, h: 0.55, fontSize: 13, italic: true, color: GRAY, fontFace: "Calibri", margin: 0 }
  );

  const rows = [
    [{ text: "Learned Shortcut Feature", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
     { text: "Effect on Prediction", options: { bold: true, color: WHITE, fill: { color: NAVY } } }],
    [{ text: "FFHQ-style colour distribution" }, { text: "Classified as Real", options: { color: GREEN, bold: true } }],
    [{ text: "Modern HDR processing" }, { text: "Classified as Fake", options: { color: RED, bold: true } }],
    [{ text: "Strong sharpening / computational photography" }, { text: "Increased false positives", options: { color: RED, bold: true } }],
    [{ text: "High colour saturation" }, { text: "Increased false positives", options: { color: RED, bold: true } }],
    [{ text: "Modern device image statistics" }, { text: "Misclassified as AI-generated", options: { color: RED, bold: true } }],
  ];
  s.addTable(rows, {
    x: 0.55, y: 2.4, w: 12.2, h: 3.0, fontSize: 13, fontFace: "Calibri", color: TEXT_DARK,
    border: { type: "solid", color: "E5E5EA", pt: 1 }, autoPage: false, colW: [7.3, 4.9],
  });

  card(s, 0.55, 5.65, 12.2, 1.15, GRAY_LIGHT);
  s.addText([
    { text: "100% accuracy ", options: { bold: true, color: GREEN } },
    { text: "on FFHQ-like images, dropping to ", options: {} },
    { text: "8.6% ", options: { bold: true, color: RED } },
    { text: "on recent real photographs — even though both datasets contain genuine human faces.", options: {} },
  ], { x: 0.85, y: 5.85, w: 11.6, h: 0.75, fontSize: 13, color: TEXT_DARK, fontFace: "Calibri", valign: "middle", margin: 0 });

  footerBrand(s, false);
  pageNum(s, 14);
}

// ============================================================
// SLIDE 15 — EXPLAINABILITY / GRAD-CAM
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  contentTitle(s, "Section 6", "Explainability — Grad-CAM");

  const capW = 5.85;
  s.addText("Correct prediction — true Fake", { x: 0.55, y: 1.6, w: capW, h: 0.35, fontSize: 12, bold: true, color: GREEN, fontFace: "Calibri", margin: 0 });
  s.addImage({ path: IMG.gcCorrect, x: 0.55, y: 2.0, w: capW, h: capW * (750 / 2250) });
  s.addText("Heatmap concentrates on the eye/nose region — the model attends to facial structure, consistent with genuine synthesis artifacts.", {
    x: 0.55, y: 2.0 + capW * (750 / 2250) + 0.15, w: capW, h: 1.0, fontSize: 10.5, color: GRAY, fontFace: "Calibri", margin: 0,
  });

  s.addText("Failure case — true Real, predicted Fake", { x: 6.9, y: 1.6, w: capW, h: 0.35, fontSize: 12, bold: true, color: RED, fontFace: "Calibri", margin: 0 });
  s.addImage({ path: IMG.gcFailure, x: 6.9, y: 2.0, w: capW, h: capW * (750 / 2250) });
  s.addText("A genuine photo misclassified as Fake — consistent with the model reacting to modern camera post-processing rather than real forgery evidence.", {
    x: 6.9, y: 2.0 + capW * (750 / 2250) + 0.15, w: capW, h: 1.0, fontSize: 10.5, color: GRAY, fontFace: "Calibri", margin: 0,
  });

  card(s, 0.55, 5.15, 12.2, 1.75, GRAY_LIGHT);
  s.addText("Reading a Grad-CAM heatmap", { x: 0.85, y: 5.32, w: 11.6, h: 0.35, fontSize: 13, bold: true, color: NAVY, fontFace: "Calibri", margin: 0 });
  const gcPoints = [
    "Brighter (red/yellow) regions contributed more to the predicted class — it visualizes where the model looked, not why in words.",
    "Verified on both a correct case and a failure case, not just successes — a useful explainability check has to show where the model goes wrong too.",
    "Correct predictions concentrate on facial structure (eyes, nose, mouth), evidence against pure background/edge shortcuts. The failure case's heatmap pattern is consistent with the model reacting to image-level properties (sharpening, HDR) rather than real forgery evidence — direct visual support for the Section 5.4 root-cause finding.",
  ];
  s.addText(
    gcPoints.map((t, i) => ({ text: t, options: { bullet: { code: "2022" }, color: TEXT_DARK, fontSize: 10.8, breakLine: i < gcPoints.length - 1, paraSpaceAfter: 6 } })),
    { x: 0.85, y: 5.68, w: 11.6, h: 1.15, fontFace: "Calibri", valign: "top", margin: 0 }
  );

  footerBrand(s, false);
  pageNum(s, 15);
}

// ============================================================
// SLIDE 16 — ROBUSTNESS & OOD TESTING
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  contentTitle(s, "Sections 6 & 7", "Robustness & Out-of-Distribution Testing");

  card(s, 0.55, 1.75, 5.95, 4.6, GRAY_LIGHT);
  s.addText("Manipulation Stress Test", { x: 0.85, y: 1.95, w: 5.4, h: 0.4, fontSize: 15, bold: true, color: NAVY, fontFace: "Calibri", margin: 0 });
  s.addText("11 corruptions: tints, brightness, contrast, blur, JPEG, resize, crop, noise", { x: 0.85, y: 2.35, w: 5.4, h: 0.35, fontSize: 10.5, italic: true, color: GRAY, fontFace: "Calibri", margin: 0 });

  card(s, 0.85, 2.85, 5.35, 1.4, GREEN_SOFT);
  s.addText("True-Real sample", { x: 1.05, y: 2.98, w: 4.9, h: 0.3, fontSize: 11.5, bold: true, color: GREEN, fontFace: "Calibri", margin: 0 });
  s.addText("Held \"Real\" correctly across all 11 manipulations — genuine robustness.", { x: 1.05, y: 3.28, w: 4.9, h: 0.9, fontSize: 10.5, color: TEXT_DARK, fontFace: "Calibri", margin: 0 });

  card(s, 0.85, 4.35, 5.35, 1.4, RED_SOFT);
  s.addText("True-Fake sample", { x: 1.05, y: 4.48, w: 4.9, h: 0.3, fontSize: 11.5, bold: true, color: RED, fontFace: "Calibri", margin: 0 });
  s.addText("Stayed incorrectly \"Real\" across all 11 — the model can be stably wrong. Robustness is not correctness.", { x: 1.05, y: 4.78, w: 4.9, h: 0.9, fontSize: 10.5, color: TEXT_DARK, fontFace: "Calibri", margin: 0 });

  card(s, 6.85, 1.75, 5.95, 4.6, ICE_SOFT);
  s.addText("Out-of-Distribution: Non-Face Images", { x: 7.15, y: 1.95, w: 5.4, h: 0.4, fontSize: 15, bold: true, color: NAVY, fontFace: "Calibri", margin: 0 });
  s.addText("10 real, license-clean CV test images — objects, animals, textures, landscapes", { x: 7.15, y: 2.35, w: 5.4, h: 0.35, fontSize: 10.5, italic: true, color: GRAY, fontFace: "Calibri", margin: 0 });

  statTile(s, 7.15, 2.85, 1.7, 1.15, "4/10", "predicted Real", NAVY);
  statTile(s, 8.95, 2.85, 1.7, 1.15, "6/10", "predicted Fake", NAVY);
  statTile(s, 10.75, 2.85, 1.7, 1.15, "86.7%", "avg. confidence", RED);

  s.addText(
    "No \"I don't know\" option — softmax always forces a confident decision. Close to a coin flip with no discernible pattern by content type. Any Real/Fake prediction on a non-face image is meaningless, not just low-confidence.",
    { x: 7.15, y: 4.2, w: 5.4, h: 1.9, fontSize: 11.5, color: TEXT_DARK, fontFace: "Calibri", margin: 0 }
  );

  footerBrand(s, false);
  pageNum(s, 16);
}

// ============================================================
// SLIDE 17 — DEPLOYMENT READINESS
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  contentTitle(s, "Section 9", "Deployment Readiness");

  const stats = [
    ["16.24 MB", "Model size on disk", NAVY],
    ["15.53 ms", "CPU latency (measured)", NAVY],
    ["64.4 /s", "CPU throughput (measured)", NAVY],
    ["332.6 MB", "Process RAM (measured)", NAVY],
  ];
  const sw = 2.7, sh = 1.3, sx0 = 0.55, sy0 = 1.85, sgap = 0.15;
  stats.forEach(([v, l, c], i) => statTile(s, sx0 + i * (sw + sgap), sy0, sw, sh, v, l, c));

  card(s, 0.55, 3.4, 5.9, 3.65, GRAY_LIGHT);
  s.addText("The model is not the bottleneck", { x: 0.85, y: 3.6, w: 5.3, h: 0.4, fontSize: 14, bold: true, color: NAVY, fontFace: "Calibri", margin: 0 });
  s.addText(
    "~140x gap between the raw forward pass (15.53 ms) and a full Grad-CAM-enabled request (~2.2 s). The overhead is Grad-CAM rendering, not classification.",
    { x: 0.85, y: 4.05, w: 5.3, h: 0.85, fontSize: 11.5, color: TEXT_DARK, fontFace: "Calibri", margin: 0 }
  );
  s.addText([
    { text: "GPU latency (est., not measured):  ", options: { bold: true } }, { text: "~7-8 ms\n", options: { color: RED } },
    { text: "GPU VRAM (est., not measured):  ", options: { bold: true } }, { text: "low hundreds of MB", options: { color: RED } },
  ], { x: 0.85, y: 5.0, w: 5.3, h: 0.9, fontSize: 11.5, fontFace: "Calibri", margin: 0, lineSpacing: 22 });

  card(s, 6.75, 3.4, 6.05, 3.65, ICE_SOFT);
  s.addText("Quantization potential", { x: 7.05, y: 3.6, w: 5.5, h: 0.4, fontSize: 14, bold: true, color: NAVY, fontFace: "Calibri", margin: 0 });
  const qb = [
    "FP16: ~22.6 MB, negligible accuracy loss",
    "INT8: ~11-12 MB, well-suited to edge/mobile",
    "Pruning/distillation: poor fit — already compact",
    "Quantizing alone won't fix latency — Grad-CAM is the real bottleneck",
  ];
  s.addText(qb.map((t, i) => ({ text: t, options: { bullet: { code: "2022" }, color: TEXT_DARK, fontSize: 11.5, breakLine: i < qb.length - 1, paraSpaceAfter: 10 } })), {
    x: 7.05, y: 4.05, w: 5.5, h: 2.85, fontFace: "Calibri", valign: "top", margin: 0,
  });

  footerBrand(s, false);
  pageNum(s, 17);
}

// ============================================================
// SLIDE 18 — KEY FINDINGS & FUTURE WORK
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  contentTitle(s, "Sections 8 & 10", "Key Findings & Future Work");

  card(s, 0.55, 1.75, 5.95, 4.65, GRAY_LIGHT);
  s.addText("Headline Findings", { x: 0.85, y: 1.95, w: 5.4, h: 0.4, fontSize: 15, bold: true, color: NAVY, fontFace: "Calibri", margin: 0 });
  const findings = [
    "99.63% in-distribution accuracy was never the problem — generalization was",
    "8.6% / 62.0% accuracy on real photos, confirmed by two independent probes",
    "Root cause identified: HDR, sharpening, saturation — not forgery signals",
    "Grad-CAM, not the model, is the deployment latency bottleneck",
    "Automated forensic report generation now built into the web app (/report)",
  ];
  s.addText(findings.map((t, i) => ({ text: t, options: { bullet: { code: "2022" }, color: TEXT_DARK, fontSize: 11.5, breakLine: i < findings.length - 1, paraSpaceAfter: 12 } })), {
    x: 0.85, y: 2.4, w: 5.4, h: 3.85, fontFace: "Calibri", valign: "top", margin: 0,
  });

  card(s, 6.75, 1.75, 6.05, 4.65, ICE_SOFT);
  s.addText("Actionable Next Steps", { x: 7.05, y: 1.95, w: 5.5, h: 0.4, fontSize: 15, bold: true, color: NAVY, fontFace: "Calibri", margin: 0 });
  s.addText("SHORT-TERM", { x: 7.05, y: 2.4, w: 5.5, h: 0.3, fontSize: 10, bold: true, color: GRAY, charSpacing: 1, fontFace: "Calibri", margin: 0 });
  const st = ["Recalibrate the decision threshold", "Make Grad-CAM optional for latency", "Run a proper demographic-fairness audit"];
  s.addText(st.map((t, i) => ({ text: t, options: { bullet: { code: "2022" }, color: TEXT_DARK, fontSize: 11, breakLine: i < st.length - 1, paraSpaceAfter: 6 } })), {
    x: 7.05, y: 2.7, w: 5.5, h: 1.2, fontFace: "Calibri", valign: "top", margin: 0,
  });
  s.addText("LONG-TERM", { x: 7.05, y: 4.0, w: 5.5, h: 0.3, fontSize: 10, bold: true, color: GRAY, charSpacing: 1, fontFace: "Calibri", margin: 0 });
  const lt = ["Targeted HDR/sharpening augmentation", "Hard-negative mining on Real-Latest failures", "Complete cross-domain model training"];
  s.addText(lt.map((t, i) => ({ text: t, options: { bullet: { code: "2022" }, color: TEXT_DARK, fontSize: 11, breakLine: i < lt.length - 1, paraSpaceAfter: 6 } })), {
    x: 7.05, y: 4.3, w: 5.5, h: 1.9, fontFace: "Calibri", valign: "top", margin: 0,
  });

  footerBrand(s, false);
  pageNum(s, 18);
}

// ============================================================
// SLIDE 19 — THANK YOU
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addShape("ellipse", { x: -2.2, y: -2.5, w: 6, h: 6, fill: { color: NAVY_DARK }, line: { type: "none" } });
  s.addShape("ellipse", { x: 10, y: 4, w: 5.5, h: 5.5, fill: { color: NAVY_DARK }, line: { type: "none" } });

  s.addText("Thank You", { x: 0.9, y: 2.9, w: 11.5, h: 1.3, fontSize: 54, bold: true, color: WHITE, fontFace: "Cambria", margin: 0 });
  s.addText("Group 11  ·  Deep Learning-Based Human Face Authenticity Detection", {
    x: 0.9, y: 4.1, w: 11, h: 0.5, fontSize: 15, color: ICE, fontFace: "Calibri", italic: true, margin: 0,
  });
  s.addShape("line", { x: 0.9, y: 4.75, w: 2.4, h: 0, line: { color: ICE, width: 1.5 } });
  s.addText("Vishakha  ·  Rohit  ·  Aman  ·  Raunak  ·  Somendu", {
    x: 0.9, y: 4.95, w: 10, h: 0.4, fontSize: 12.5, color: WHITE, fontFace: "Calibri", margin: 0,
  });
}

pres.writeFile({ fileName: "C:/Users/Vishakha.Roy/Downloads/Deepfake/Group-11-DS-and-AI-Lab-Project/doc/Milestone-5/ppt/Milestone5_Presentation.pptx" })
  .then(() => console.log("Deck written."))
  .catch((e) => { console.error(e); process.exit(1); });
