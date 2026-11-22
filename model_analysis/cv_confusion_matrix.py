
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "research_dataset_clean.csv"

df = pd.read_csv(DATASET_PATH)
df = df.dropna(subset=["text", "primary_department"])

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
    ("tfidf", TfidfVectorizer()),
    ("svm", LinearSVC())
])

# Each complaint receives a prediction from a model
# that did not train on that complaint.
predictions = cross_val_predict(
    model,
    X,
    y,
    cv=cv
)

labels = sorted(y.unique())

print("\nCONFUSION MATRIX")
print("Department order:", labels)
print(confusion_matrix(y, predictions, labels=labels))

print("\nDEPARTMENT-WISE CLASSIFICATION REPORT")
print(classification_report(
    y,
    predictions,
    labels=labels,
    zero_division=0
))

output = pd.DataFrame({
    "text": X.values,
    "actual": y.values,
    "predicted": predictions,
    "correct": y.values == predictions
})
output.to_csv(
    BASE_DIR / "model_analysis" / "cv_predictions.csv",
    index=False
)

print("\nSaved: model_analysis/cv_predictions.csv")
