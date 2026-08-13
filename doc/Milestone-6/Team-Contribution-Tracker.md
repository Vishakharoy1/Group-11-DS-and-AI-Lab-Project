# Team Contribution Tracker - Milestone 6

**Project:** Deep Learning-Based Human Face Authenticity Detection

This document tracks the work assigned for Milestone 6 (Final Submission &
Deployment), per the task division agreed in `Group11_M6_TaskDivision.docx`.
Deadline: 13 August 2026.

Tasks are divided based on each member's demonstrated area of ownership from
M4/M5 and the specific gaps raised in the M5 review meeting. Each person
continues the thread of work they already built, so no context is lost and
each fix lands with the person best positioned to make it.

Three PRIORITY issues came out of the M5 review and are called out below:
(1) **domain shift / shortcut learning** — 99.63% in-distribution accuracy
collapses to 8.6% on real smartphone photos ("Real-Latest"), because the
model keys on HDR tone-mapping, saturation, and sharpening rather than
authenticity cues; (2) **latency bottleneck** — `/predict` takes seconds to
minutes end-to-end because Grad-CAM runs on every request; (3) **metric/
ethics gap** — ROC-AUC (0.5856) was only computed on a 50-image sample, not
the full 2,401-image held-out set across all 4 modules, and dataset splits
were never audited for demographic bias.

M6 also requires seven final deliverables: presentation, technical report,
non-technical report, user guide, developer guide + code, a stable
deployment, and a contribution summary.

## 1. Vishakha - Pipeline & Presentation Lead

*M5 focus area: preprocessing verification, robustness testing,
presentation, README. M5 review feedback: N/A — continuation of
coordination role.*

### Assigned Tasks for Milestone 6

- Compile and finalize the **Final Presentation (Deliverable 1)** —
  assemble outputs from all members into one deck covering objectives,
  methodology, results, process, and key takeaways for both technical and
  non-technical audiences.
- Own **Project Deployment stability (Deliverable 6)** — dedicated
  instruction page with preset examples and custom upload option; final
  end-to-end verification once Somendu's Grad-CAM fix is merged.
- Compile the final **Contribution Summary (Deliverable 7)** — document
  each member's involvement across M1–M6.
- Re-verify UI backend preprocessing still matches the training notebook
  after the Stage 3 / CelebA-HD retraining update.
- **Developer guide and code** — documentation enabling another developer
  to replicate the project setup: setup steps, dependencies, all code
  scripts, configurations, and implementation notes.

---

## 2. Rohit - Training Stability

*M5 focus area: evaluation, confusion matrix, ROC curve, latency
measurement. M5 review feedback: ROC-AUC was only computed on a 50-image
sample (0.5856) instead of the full held-out set.*

### Assigned Tasks for Milestone 6

- **[PRIORITY]** Compute a proper threshold-independent ROC-AUC on the
  full 2,401-image held-out test set, across all 4 modules — replacing
  the 50-image sample metric (0.5856).
- Regenerate confusion matrix, precision/recall, ROC curve, and
  precision-recall curve plots on the full test set for each module.
- Lead and compile the **Final Technical Report (Deliverable 2)** — full
  M1→M6 write-up: design, pipeline, tools, results, and how each
  challenge (shortcut learning, preprocessing mismatch, domain shift) was
  addressed.
- Improve the overall UI and screen layout — two separate pages (Main
  Model showcasing the primary deepfake detection model and its results;
  Cross-Domain Model showcasing the cross-domain model and its
  evaluation/results), plus an Analysis Report section similar to the
  existing forensic report, presenting model results and analysis in a
  clear, structured format.

---

## 3. Aman - Preprocessing & Transfer Learning

*M5 focus area: 3-stage transfer learning documentation, pipeline
optimisation. M5 review feedback: N/A — documentation continuation.*

### Assigned Tasks for Milestone 6

- Write the **Non-Technical Report (Deliverable 3)** — purpose, impact,
  and plain-language functionality for a general audience.

---

## 4. Raunak - Dataset & Bias Analysis

*M5 focus area: shortcut-learning root cause, dataset composition audit.
M5 review feedback: model collapses to 8.6% accuracy on "Real-Latest"
smartphone photos due to shortcut learning on HDR/sharpening artifacts;
dataset splits never audited for demographic bias.*

### Assigned Tasks for Milestone 6

- **[PRIORITY]** Document root cause and future-work plan for domain
  shift / shortcut learning — extend the M5 root-cause analysis to
  explain the 8.6% "Real-Latest" collapse and the model's reliance on
  HDR tone-mapping, saturation, and sharpening artifacts.
- Write the ethical/demographic bias limitations note — acknowledge
  dataset splits were never audited for skin tone, age, gender, or
  ethnicity, and scope what a future audit would require.
- Contribute the challenges/limitations findings to the Technical Report
  (feeds Rohit's Deliverable 2).

---

## 5. Somendu - Explainability & Optimisation

*M5 focus area: Grad-CAM implementation, robustness/interpretability. M5
review feedback: `/predict` takes seconds to minutes because Grad-CAM
runs on every request.*

### Assigned Tasks for Milestone 6

- **[PRIORITY]** Fix the latency bottleneck — convert Grad-CAM generation
  into a manual trigger button (or async job) so `/predict` returns fast
  for plain classification, with Grad-CAM available on demand.
- Verify raw MobileNetV3 forward-pass time vs. Grad-CAM generation time
  separately, and confirm the fix resolves the seconds-to-minutes
  end-to-end delay.
- Write the **User Guide (Deliverable 4)** — steps, screenshots, and
  examples using the deployed app, once the latency fix is live.
- **Developer guide and code** — documentation enabling another developer
  to replicate the project setup: setup steps, dependencies, all code
  scripts, configurations, and implementation notes.

---

## Team Declaration

We certify that all team members have actively contributed to the
preparation of Milestone 6, per the task division recorded in
`Group11_M6_TaskDivision.docx`. Each member has reviewed the contents of
this document and agrees with the assigned scope.

| Team Member | Role | Status | Signature |
| --- | --- | --- | --- |
| Vishakha | Pipeline & Presentation Lead | Per task division — pending own review/sign-off | Vishakha |
| Rohit | Training Stability | Per task division — pending own review/sign-off | Rohit |
| Aman | Preprocessing & Transfer Learning | Per task division — pending own review/sign-off | Aman |
| Raunak | Dataset & Bias Analysis | Per task division — pending own review/sign-off | Raunak |
| Somendu | Explainability & Optimisation | Per task division — pending own review/sign-off | Somendu |
