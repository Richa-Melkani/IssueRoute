
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    roc_auc_score
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET = BASE_DIR / "dataset" / "research_dataset_clean.csv"

OUT = BASE_DIR / "outputs" / "single_department_evaluation"
OUT.mkdir(parents=True, exist_ok=True)

# Load single-department complaints only
df = pd.read_csv(DATASET)
df = df[df["complaint_type"] == "single"].copy()
df = df.dropna(subset=["text", "primary_department"])
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]

X = df["text"]
y = df["primary_department"]

labels = sorted(y.unique())

print("Total complaints:", len(df))
print("Departments:", labels)

# Five-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("svm", LinearSVC())
])

predictions = cross_val_predict(model, X, y, cv=cv)
scores = cross_val_predict(
    model, X, y, cv=cv, method="decision_function"
)

# 1. Confusion matrix
cm = confusion_matrix(y, predictions, labels=labels)

fig, ax = plt.subplots(figsize=(11, 8))
image = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticklabels(labels)
ax.set_xlabel("Predicted Department")
ax.set_ylabel("Actual Department")
ax.set_title("Single Department - Confusion Matrix (5-Fold CV)")

for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, cm[i, j], ha="center", va="center")

fig.colorbar(image)
fig.tight_layout()
fig.savefig(OUT / "confusion_matrix.png", dpi=300)
plt.close(fig)

# 2. Classification report
report = classification_report(
    y, predictions,
    labels=labels,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).transpose().to_csv(
    OUT / "classification_report.csv"
)

# 3. Overall evaluation metrics
metrics = {
    "Accuracy": accuracy_score(y, predictions),
    "Macro Precision": precision_score(
        y, predictions, average="macro", zero_division=0
    ),
    "Macro Recall": recall_score(
        y, predictions, average="macro", zero_division=0
    ),
    "Macro F1": f1_score(
        y, predictions, average="macro", zero_division=0
    ),
    "Weighted Precision": precision_score(
        y, predictions, average="weighted", zero_division=0
    ),
    "Weighted Recall": recall_score(
        y, predictions, average="weighted", zero_division=0
    ),
    "Weighted F1": f1_score(
        y, predictions, average="weighted", zero_division=0
    )
}

# 4. Multiclass ROC and AUC (One-vs-Rest)
y_binary = label_binarize(y, classes=labels)

macro_auc = roc_auc_score(
    y_binary, scores, average="macro"
)
micro_auc = roc_auc_score(
    y_binary, scores, average="micro"
)

metrics["Macro ROC-AUC"] = macro_auc
metrics["Micro ROC-AUC"] = micro_auc

fig, ax = plt.subplots(figsize=(9, 7))
roc_results = []

for i, department in enumerate(labels):
    fpr, tpr, _ = roc_curve(y_binary[:, i], scores[:, i])
    class_auc = auc(fpr, tpr)

    ax.plot(
        fpr, tpr,
        label=f"{department} (AUC = {class_auc:.3f})"
    )

    roc_results.append({
        "Department": department,
        "AUC": class_auc
    })

ax.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Single Department - ROC Curves (One-vs-Rest)")
ax.legend(loc="lower right", fontsize=8)
ax.grid()
fig.tight_layout()
fig.savefig(OUT / "roc_curve.png", dpi=300)
plt.close(fig)

pd.DataFrame(roc_results).to_csv(
    OUT / "department_auc.csv", index=False
)

pd.DataFrame(
    [{"Metric": k, "Score": v} for k, v in metrics.items()]
).to_csv(OUT / "overall_metrics.csv", index=False)

# 5. Save predictions
pd.DataFrame({
    "complaint": X.values,
    "actual": y.values,
    "predicted": predictions,
    "correct": y.values == predictions
}).to_csv(OUT / "predictions.csv", index=False)

print("\nOVERALL METRICS")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nAll files saved in:", OUT)

