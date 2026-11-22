
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    BASE_DIR / "dataset" / "research_dataset_clean.csv"
)

df = df.dropna(subset=[
    "text", "primary_department", "language"
])

df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]

X = df["text"]
y = df["primary_department"]

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

model = Pipeline([
    ("features", FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 1)
        )),
        ("char", TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=1
        ))
    ])),
    ("svm", LinearSVC())
])

predictions = cross_val_predict(
    model, X, y, cv=cv
)

results = df.copy()
results["predicted_department"] = predictions
results["correct"] = (
    results["primary_department"] ==
    results["predicted_department"]
)

hindi_errors = results[
    (results["language"] == "Hindi") &
    (results["correct"] == False)
]

print("\nTOTAL HINDI ERRORS:", len(hindi_errors))

for _, row in hindi_errors.iterrows():
    print("\nComplaint:", row["text"])
    print("Actual:", row["primary_department"])
    print("Predicted:", row["predicted_department"])
    print("-" * 50)

output_path = (
    BASE_DIR / "model_analysis" /
    "hindi_misclassified_complaints.csv"
)

hindi_errors.to_csv(output_path, index=False)

print("\nSaved:", output_path)
