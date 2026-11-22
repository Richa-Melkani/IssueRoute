
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    BASE_DIR / "dataset" / "research_dataset_clean.csv"
)

df = df.dropna(subset=["text", "primary_department"])
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]

X = df["text"]
y = df["primary_department"]
languages = df["language"]

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
    model,
    X,
    y,
    cv=cv
)

departments = sorted(y.unique())

for language in ["Hindi", "English", "Hinglish"]:

    mask = languages == language

    print("\n" + "=" * 50)
    print("LANGUAGE:", language)
    print("=" * 50)

    print(classification_report(
        y[mask],
        predictions[mask],
        labels=departments,
        zero_division=0
    ))

print("\nCompleted department-wise analysis.")
