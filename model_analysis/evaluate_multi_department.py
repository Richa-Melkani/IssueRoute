import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score, hamming_loss,
    precision_score, recall_score, f1_score,
    classification_report, multilabel_confusion_matrix,
    roc_curve, roc_auc_score
)

# Paths
DATA_PATH = "dataset/research_dataset_clean.csv"
OUTPUT_DIR = "outputs/multi_department_evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and filter multi-department complaints
df = pd.read_csv(DATA_PATH)
df = df[df["complaint_type"].astype(str).str.lower() == "multi"].copy()
df = df.dropna(subset=["text", "primary_department"])

def clean_label(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    if value.lower() in ["", "none", "nan", "null"]:
        return None
    return value

texts = df["text"].astype(str).tolist()

labels = []
for _, row in df.iterrows():
    row_labels = set()
    primary = clean_label(row["primary_department"])
    secondary = clean_label(row["secondary_department"])

    if primary:
        row_labels.add(primary)
    if secondary:
        row_labels.add(secondary)

    labels.append(sorted(row_labels))

# Convert department labels into a binary matrix
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(labels)
classes = mlb.classes_

print("Multi-department complaints:", len(texts))
print("Departments:", list(classes))

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

y_true = y.copy()
y_pred = np.zeros_like(y)
scores = np.zeros(y.shape, dtype=float)

for fold, (train_idx, test_idx) in enumerate(kf.split(texts), start=1):
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=10000
        )),
        ("classifier", OneVsRestClassifier(
            LinearSVC(class_weight="balanced")
        ))
    ])

    X_train = [texts[i] for i in train_idx]
    X_test = [texts[i] for i in test_idx]

    model.fit(X_train, y[train_idx])
    y_pred[test_idx] = model.predict(X_test)
    scores[test_idx] = model.decision_function(X_test)

    print(f"Completed fold {fold}/5")

# Overall metrics
metrics = {
    "Exact Match Accuracy": accuracy_score(y_true, y_pred),
    "Hamming Loss": hamming_loss(y_true, y_pred),
    "Macro Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
    "Macro Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    "Micro Precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
    "Micro Recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
    "Micro F1": f1_score(y_true, y_pred, average="micro", zero_division=0)
}

# Classification report
report = classification_report(
    y_true, y_pred,
    target_names=classes,
    output_dict=True,
    zero_division=0
)
pd.DataFrame(report).transpose().to_csv(
    os.path.join(OUTPUT_DIR, "classification_report.csv")
)

# Per-department ROC-AUC
auc_rows = []
valid_aucs = []

plt.figure(figsize=(9, 7))

for i, department in enumerate(classes):
    if len(np.unique(y_true[:, i])) < 2:
        auc_rows.append({
            "Department": department,
            "ROC-AUC": np.nan
        })
        continue

    auc = roc_auc_score(y_true[:, i], scores[:, i])
    valid_aucs.append(auc)
    auc_rows.append({
        "Department": department,
        "ROC-AUC": auc
    })

    fpr, tpr, _ = roc_curve(y_true[:, i], scores[:, i])
    plt.plot(fpr, tpr, label=f"{department} (AUC={auc:.3f})")

if valid_aucs:
    metrics["Macro ROC-AUC"] = float(np.mean(valid_aucs))

if y_true.sum() > 0:
    try:
        metrics["Micro ROC-AUC"] = roc_auc_score(
            y_true.ravel(), scores.ravel()
        )
    except ValueError:
        metrics["Micro ROC-AUC"] = np.nan

plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-Department ROC Curves")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=300)
plt.close()

pd.DataFrame(auc_rows).to_csv(
    os.path.join(OUTPUT_DIR, "department_auc.csv"), index=False
)

# Per-department confusion matrices
cms = multilabel_confusion_matrix(y_true, y_pred)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, department in enumerate(classes):
    cm = cms[i]
    axes[i].imshow(cm, cmap="Blues")
    axes[i].set_title(department)
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")
    axes[i].set_xticks([0, 1])
    axes[i].set_yticks([0, 1])

    for r in range(2):
        for c in range(2):
            axes[i].text(c, r, str(cm[r, c]),
                         ha="center", va="center")

for j in range(len(classes), len(axes)):
    axes[j].axis("off")

fig.suptitle("Multi-Department Confusion Matrices")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrices.png"),
    dpi=300
)
plt.close()

# Save overall metrics
pd.DataFrame(
    list(metrics.items()),
    columns=["Metric", "Score"]
).to_csv(os.path.join(OUTPUT_DIR, "overall_metrics.csv"), index=False)

# Save actual and predicted departments
prediction_rows = []
for i in range(len(texts)):
    actual = [classes[j] for j in range(len(classes)) if y_true[i, j] == 1]
    predicted = [classes[j] for j in range(len(classes)) if y_pred[i, j] == 1]

    prediction_rows.append({
        "Complaint": texts[i],
        "Actual Departments": "; ".join(actual),
        "Predicted Departments": "; ".join(predicted)
    })

pd.DataFrame(prediction_rows).to_csv(
    os.path.join(OUTPUT_DIR, "predictions.csv"), index=False
)

print("\nMULTI-DEPARTMENT EVALUATION RESULTS")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}" if pd.notna(value) else f"{name}: N/A")

print("\nAll files saved in:", OUTPUT_DIR)
