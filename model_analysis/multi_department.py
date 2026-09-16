import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "dataset" / "research_dataset.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "multi_department_model.pkl"


# Load research dataset
df = pd.read_csv(DATASET_PATH)


# Keep only multi-department complaints
multi_df = df[df["complaint_type"] == "multi"].copy()


# Create department labels
multi_df["departments"] = multi_df.apply(
    lambda row: [
        row["primary_department"],
        row["secondary_department"]
    ],
    axis=1
)


# Complaint text
X = multi_df["text"]


# Convert department labels into multi-label format
mlb = MultiLabelBinarizer()

y = mlb.fit_transform(multi_df["departments"])


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# TF-IDF + One-vs-Rest Linear SVM
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer()
    ),
    (
        "svm",
        OneVsRestClassifier(
            LinearSVC()
        )
    )
])


# Train model
model.fit(X_train, y_train)


# Save model and label encoder together
model_data = {
    "model": model,
    "mlb": mlb
}


joblib.dump(
    model_data,
    MODEL_PATH
)


print("Multi-department model saved successfully:")
print(MODEL_PATH)


# Test prediction
complaint = (
    "My hostel room has no water supply "
    "and my semester fee payment is failing."
)


prediction = model.predict([complaint])

departments = mlb.inverse_transform(prediction)

print("Complaint:", complaint)
print("Predicted Departments:", list(departments[0]))