# ============================================================
# Real ROC-AUC + PR Curve for mobilenetv3_best.pth, on the
# ACTUAL 2,401-image held-out test set.
#
# Paste this as a NEW cell in final-mobilenet (1).ipynb, directly
# AFTER Cell 8 (Final Evaluation) has already run successfully -
# it reuses `model`, `test_loader`, `DEVICE` from that cell, so
# there's nothing new to load or configure.
#
# No CUDA-specific calls (no .synchronize(), no memory stats) -
# this only needs whatever device Cell 8 already ran on.
# ============================================================

import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

model.eval()

all_labels = []
all_probs_fake = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE, non_blocking=True)
        with autocast():
            outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # P(Fake)
        all_labels.extend(labels.numpy())
        all_probs_fake.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_probs_fake = np.array(all_probs_fake)
print(f"Collected {len(all_labels)} predictions from test_loader.")

# ------------------------------------------------------------
# ROC curve + AUC
# ------------------------------------------------------------
fpr, tpr, _ = roc_curve(all_labels, all_probs_fake)
roc_auc = auc(fpr, tpr)
print(f"ROC-AUC (real held-out test set): {roc_auc:.4f}")

plt.figure(figsize=(5.5, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — mobilenetv3_best.pth (real held-out test set)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("/kaggle/working/roc_curve_real_testset.png", dpi=150)
plt.show()

# ------------------------------------------------------------
# Precision-Recall curve + Average Precision
# ------------------------------------------------------------
precision, recall, _ = precision_recall_curve(all_labels, all_probs_fake)
ap = average_precision_score(all_labels, all_probs_fake)
print(f"Average Precision (PR-AUC): {ap:.4f}")

plt.figure(figsize=(5.5, 5))
plt.plot(recall, precision, label=f"PR curve (AP = {ap:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve — mobilenetv3_best.pth (real held-out test set)")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("/kaggle/working/pr_curve_real_testset.png", dpi=150)
plt.show()

# ------------------------------------------------------------
# Save results CSV
# ------------------------------------------------------------
import csv
with open("/kaggle/working/roc_auc_real_testset.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Metric", "Value"])
    w.writerow(["N_test_images", len(all_labels)])
    w.writerow(["ROC-AUC", f"{roc_auc:.4f}"])
    w.writerow(["Average_Precision_PR_AUC", f"{ap:.4f}"])
print("\nSaved: roc_curve_real_testset.png, pr_curve_real_testset.png, roc_auc_real_testset.csv")
print("Download all three from the Kaggle Output tab and send them back.")
