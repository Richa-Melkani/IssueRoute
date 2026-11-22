
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

dataset = pd.read_csv(
    BASE_DIR / "dataset" / "research_dataset_clean.csv"
)

errors = pd.read_csv(
    BASE_DIR / "model_analysis" /
    "hindi_misclassified_complaints.csv"
)

# Keep Hindi complaints only
hindi = dataset[
    dataset["language"] == "Hindi"
].copy()

# Merge predictions with original complaint details
results = errors.merge(
    hindi[
        [
            "text",
            "primary_department",
            "secondary_department",
            "complaint_type"
        ]
    ],
    on=["text", "primary_department"],
    how="left",
    suffixes=("", "_dataset")
)

def check_secondary(row):
    secondary = row["secondary_department"]

    if pd.notna(secondary):
        return row["predicted_department"] == secondary

    return False

results["valid_secondary_prediction"] = results.apply(
    check_secondary,
    axis=1
)

print("\n========== ERROR BREAKDOWN ==========")

print("\nErrors by complaint type:")
print(results["complaint_type"].value_counts())

print("\nPredictions matching secondary department:")
print(
    results["valid_secondary_prediction"].sum()
)

print("\n========== SINGLE COMPLAINT ERRORS ==========")

print(
    results[
        results["complaint_type"] == "single"
    ][
        ["text", "primary_department", "predicted_department"]
    ].to_string(index=False)
)

print("\n========== MULTI COMPLAINT ERRORS ==========")

print(
    results[
        results["complaint_type"] == "multi"
    ][
        [
            "text",
            "primary_department",
            "secondary_department",
            "predicted_department",
            "valid_secondary_prediction"
        ]
    ].to_string(index=False)
)

print("\n========== UNKNOWN COMPLAINT ERRORS ==========")

print(
    results[
        results["complaint_type"] == "unknown"
    ][
        ["text", "primary_department", "predicted_department"]
    ].to_string(index=False)
)

results.to_csv(
    BASE_DIR / "model_analysis" /
    "hindi_routing_error_breakdown.csv",
    index=False
)

print("\nAnalysis saved successfully.")
