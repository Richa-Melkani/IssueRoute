
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# 1. Load dataset
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR / "dataset" / "research_dataset_clean.csv"
)

df = pd.read_csv(DATASET_PATH)
df = df.dropna(subset=["text", "primary_department"])

df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]

X = df["text"]
y = df["primary_department"]
languages = df["language"]

# 2. Same 5-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 3. Define feature extraction methods
feature_methods = {

    "Word TF-IDF": TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 1)
    ),

    "Character TF-IDF": TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=1
    ),

    "Combined Word + Character": FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 1)
            )
        ),
        (
            "character",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
                min_df=1
            )
        )
    ])
}

overall_results = []
language_results = []

# 4. Train and evaluate
for method_name, vectorizer in feature_methods.items():

    model = Pipeline([
        ("tfidf", vectorizer),
        ("svm", LinearSVC())
    ])

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv
    )

    # Overall performance
    overall_results.append({
        "Method": method_name,
        "Accuracy": accuracy_score(y, predictions),
        "Macro Precision": precision_score(
            y, predictions,
            average="macro",
            zero_division=0
        ),
        "Macro Recall": recall_score(
            y, predictions,
            average="macro",
            zero_division=0
        ),
        "Macro F1": f1_score(
            y, predictions,
            average="macro",
            zero_division=0
        )
    })

    # Language-wise performance
    for language in sorted(languages.unique()):

        mask = languages == language

        language_results.append({
            "Method": method_name,
            "Language": language,
            "Accuracy": accuracy_score(
                y[mask],
                predictions[mask]
            ),
            "Macro F1": f1_score(
                y[mask],
                predictions[mask],
                average="macro",
                zero_division=0
            )
        })

    print("\nCompleted:", method_name)

# 5. Display overall results
overall_df = pd.DataFrame(overall_results)

print("\n========== OVERALL RESULTS ==========")
print(overall_df.round(4).to_string(index=False))

# 6. Display language-wise results
language_df = pd.DataFrame(language_results)

print("\n========== LANGUAGE-WISE RESULTS ==========")
print(language_df.round(4).to_string(index=False))

# 7. Save results
overall_df.to_csv(
    BASE_DIR / "model_analysis" /
    "tfidf_overall_comparison.csv",
    index=False
)

language_df.to_csv(
    BASE_DIR / "model_analysis" /
    "tfidf_language_comparison.csv",
    index=False
)

print("\nSaved both comparison CSV files.")
