
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

predictions = pd.read_csv(
    BASE_DIR / "model_analysis" / "cv_predictions.csv"
)

dataset = pd.read_csv(
    BASE_DIR / "dataset" / "research_dataset_clean.csv"
)

# Add complaint type, language and secondary department
data = predictions.merge(
    dataset[[
        "text",
        "complaint_type",
        "language",
        "secondary_department"
    ]],
    on="text",
    how="left"
)

wrong = data[data["correct"] == False].copy()

print("\nERRORS BY COMPLAINT TYPE")
print(wrong["complaint_type"].value_counts(dropna=False))

print("\nERRORS BY LANGUAGE")
print(wrong["language"].value_counts(dropna=False))

print("\nERRORS BY ACTUAL DEPARTMENT")
print(wrong["actual"].value_counts())

print("\nSINGLE VS MULTI-DEPARTMENT ERRORS")

print(
    wrong["complaint_type"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nMULTI-DEPARTMENT MISCLASSIFICATIONS")

multi_errors = wrong[
    wrong["complaint_type"].astype(str).str.lower().str.contains("multi")
]

print(
    multi_errors[
        ["text", "actual", "predicted", "secondary_department", "language"]
    ].to_string(index=False)
)

wrong.to_csv(
    BASE_DIR / "model_analysis" / "detailed_errors.csv",
    index=False
)



print("\nERROR RATE BY COMPLAINT TYPE")

type_summary = (
    data.groupby("complaint_type")
    .agg(
        total=("correct", "count"),
        incorrect=("correct", lambda x: (~x).sum())
    )
)

type_summary["error_rate"] = (
    type_summary["incorrect"] / type_summary["total"] * 100
).round(2)

print(type_summary)


print("\nERROR RATE BY LANGUAGE")

language_summary = (
    data.groupby("language")
    .agg(
        total=("correct", "count"),
        incorrect=("correct", lambda x: (~x).sum())
    )
)

language_summary["error_rate"] = (
    language_summary["incorrect"] /
    language_summary["total"] * 100
).round(2)

print(language_summary)


from sklearn.metrics import accuracy_score, f1_score

print("\nLANGUAGE-WISE DEPARTMENT DISTRIBUTION")

print(pd.crosstab(data["language"], data["actual"]))

print("\nLANGUAGE-WISE MODEL PERFORMANCE")

for language, group in data.groupby("language"):

    labels = sorted(group["actual"].unique())

    accuracy = accuracy_score(
        group["actual"],
        group["predicted"]
    )

    macro_f1 = f1_score(
        group["actual"],
        group["predicted"],
        labels=labels,
        average="macro",
        zero_division=0
    )

    print(f"\n{language}")
    print("Total complaints:", len(group))
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Macro F1:", round(macro_f1 * 100, 2), "%")


from sklearn.metrics import classification_report, confusion_matrix

all_departments = [
    "Academics",
    "Administration",
    "Fees / Finance",
    "Hostels",
    "IT Support",
    "Maintenance",
    "Security / Discipline",
    "Transport",
    "Unknown"
]

for language in ["Hindi", "English", "Hinglish"]:

    group = data[data["language"] == language]

    print("\n" + "=" * 50)
    print("LANGUAGE:", language)
    print("=" * 50)

    print("\nDEPARTMENT-WISE CLASSIFICATION REPORT")

    print(classification_report(
        group["actual"],
        group["predicted"],
        labels=all_departments,
        zero_division=0
    ))

    print("\nCONFUSION MATRIX")
    print(confusion_matrix(
        group["actual"],
        group["predicted"],
        labels=all_departments
    ))

print("\nFULL LANGUAGE-WISE DEPARTMENT DISTRIBUTION")
print(pd.crosstab(data["language"], data["actual"]).to_string())


print("\nSaved: model_analysis/detailed_errors.csv")
