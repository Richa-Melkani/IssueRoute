
import pandas as pd
from pathlib import Path

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    f1_score,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    BASE_DIR / "dataset" / "research_dataset_clean.csv"
)

df = df[df["complaint_type"] == "multi"].copy()
df = df.dropna(subset=["text", "primary_department"])

df["departments"] = df.apply(
    lambda row: [
        dept for dept in [
            row["primary_department"],
            row["secondary_department"]
        ]
        if pd.notna(dept) and str(dept).strip()
    ],
    axis=1
)

X = df["text"]

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["departments"])

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
("classifier", OneVsRestClassifier(
    LinearSVC(class_weight="balanced")
))
])

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

predictions = cross_val_predict(
    model,
    X,
    y,
    cv=cv
)

def show_metrics(name, actual, predicted):

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print("Samples:", len(actual))

    print(
        "Exact Match Accuracy:",
        round(accuracy_score(actual, predicted), 4)
    )

    print(
        "Hamming Loss:",
        round(hamming_loss(actual, predicted), 4)
    )

    print(
        "Macro F1:",
        round(f1_score(
            actual,
            predicted,
            average="macro",
            zero_division=0
        ), 4)
    )

    print(
        "Micro F1:",
        round(f1_score(
            actual,
            predicted,
            average="micro",
            zero_division=0
        ), 4)
    )

    print("\nDepartment-wise report:")

    print(classification_report(
        actual,
        predicted,
        target_names=mlb.classes_,
        zero_division=0
    ))

# All multi-department complaints
show_metrics(
    "ALL MULTI-DEPARTMENT COMPLAINTS",
    y,
    predictions
)

# Hindi multi-department complaints
hindi_mask = (
    df["language"].to_numpy() == "Hindi"
)

show_metrics(
    "HINDI MULTI-DEPARTMENT COMPLAINTS",
    y[hindi_mask],
    predictions[hindi_mask]
)

from sklearn.metrics import multilabel_confusion_matrix

def department_prediction_analysis(name, actual, predicted):

    print("\n==========", name, "==========")

    matrices = multilabel_confusion_matrix(
        actual,
        predicted
    )

    for department, matrix in zip(mlb.classes_, matrices):

        tn, fp, fn, tp = matrix.ravel()

        print(f"\nDepartment: {department}")
        print("Actual positive:", tp + fn)
        print("Predicted positive:", tp + fp)
        print("True positives:", tp)
        print("False positives:", fp)
        print("False negatives:", fn)


department_prediction_analysis(
    "ALL LANGUAGES",
    y,
    predictions
)

department_prediction_analysis(
    "HINDI",
    y[hindi_mask],
    predictions[hindi_mask]
)

print("\nAnalysis completed.")
