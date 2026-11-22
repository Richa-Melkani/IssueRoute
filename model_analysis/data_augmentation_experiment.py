import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score, hamming_loss,
    precision_score, recall_score, f1_score
)

DATA_PATH = "dataset/research_dataset_clean.csv"
OUTPUT_DIR = "outputs/data_augmentation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    current = {
        x for x in [
            clean_label(row["primary_department"]),
            clean_label(row["secondary_department"])
        ] if x
    }
    labels.append(sorted(current))

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(labels)

def augment_text(text, rng, drop_probability=0.10):
    """Randomly remove a small proportion of words."""
    words = text.split()

    if len(words) < 4:
        return text

    kept = [word for word in words if rng.random() > drop_probability]

    if not kept:
        kept = [rng.choice(words)]

    return " ".join(kept)

def create_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=10000
        )),
        ("classifier", OneVsRestClassifier(
            LinearSVC(class_weight="balanced")
        ))
    ])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = {
    "Original": np.zeros_like(y),
    "Augmented": np.zeros_like(y)
}

for fold, (train_idx, test_idx) in enumerate(kf.split(texts), start=1):
    X_train = [texts[i] for i in train_idx]
    X_test = [texts[i] for i in test_idx]
    y_train = y[train_idx]

    # Baseline: original training data only
    baseline = create_model()
    baseline.fit(X_train, y_train)
    results["Original"][test_idx] = baseline.predict(X_test)

    # Augmentation: create one modified copy per training complaint
    rng = random.Random(100 + fold)
    augmented_texts = [
        augment_text(text, rng) for text in X_train
    ]

    X_train_augmented = X_train + augmented_texts
    y_train_augmented = np.vstack([y_train, y_train])

    augmented_model = create_model()
    augmented_model.fit(X_train_augmented, y_train_augmented)
    results["Augmented"][test_idx] = augmented_model.predict(X_test)

    print(f"Completed fold {fold}/5")

# Calculate comparable metrics
rows = []

for model_name, predictions in results.items():
    rows.append({
        "Model": model_name,
        "Exact Match Accuracy": accuracy_score(y, predictions),
        "Hamming Loss": hamming_loss(y, predictions),
        "Macro Precision": precision_score(
            y, predictions, average="macro", zero_division=0
        ),
        "Macro Recall": recall_score(
            y, predictions, average="macro", zero_division=0
        ),
        "Macro F1": f1_score(
            y, predictions, average="macro", zero_division=0
        ),
        "Micro Precision": precision_score(
            y, predictions, average="micro", zero_division=0
        ),
        "Micro Recall": recall_score(
            y, predictions, average="micro", zero_division=0
        ),
        "Micro F1": f1_score(
            y, predictions, average="micro", zero_division=0
        )
    })

comparison = pd.DataFrame(rows)
comparison.to_csv(
    os.path.join(OUTPUT_DIR, "augmentation_comparison.csv"),
    index=False
)

print("\nDATA AUGMENTATION COMPARISON")
print(comparison.to_string(index=False))
print("\nSaved:", OUTPUT_DIR)