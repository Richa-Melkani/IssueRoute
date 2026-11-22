
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "research_dataset_clean.csv"

df = pd.read_csv(DATASET_PATH)
df = df.dropna(subset=["text", "primary_department"])

df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]

X = df["text"]
y = df["primary_department"]

# Five-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "Linear SVM": LinearSVC(),
    "Logistic Regression": LogisticRegression(max_iter=2000)
}

scoring = {
    "accuracy": "accuracy",
    "macro_f1": make_scorer(f1_score, average="macro", zero_division=0),
    "macro_precision": make_scorer(
        precision_score, average="macro", zero_division=0
    ),
    "macro_recall": make_scorer(
        recall_score, average="macro", zero_division=0
    )
}

results = []

for name, classifier in models.items():

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", classifier)
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    print(f"\n========== {name} ==========")

    result = {"Model": name}

    for metric in scoring:
        values = scores[f"test_{metric}"]
        mean_score = values.mean()
        std_score = values.std()

        result[f"{metric}_mean"] = mean_score
        result[f"{metric}_std"] = std_score

        print(
            f"{metric}: {mean_score:.4f} "
            f"+/- {std_score:.4f}"
        )

    results.append(result)

results_df = pd.DataFrame(results)

output_path = (
    BASE_DIR / "model_analysis" / "cross_validation_results.csv"
)

results_df.to_csv(output_path, index=False)

print("\n========== FINAL CROSS-VALIDATION RESULTS ==========")
print(results_df.to_string(index=False))
print("\nSaved to:", output_path)
