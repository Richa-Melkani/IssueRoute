
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "research_dataset.csv"

df = pd.read_csv(DATASET_PATH)

# Normalize text only for checking
df["normalized_text"] = (
    df["text"].astype(str).str.strip().str.lower()
)

print("\n1. Exact duplicate records:")
print(
    df[df.duplicated(keep=False)]
    .sort_values("text")
    .to_string(index=False)
)

print("\n2. Duplicate complaint texts:")
duplicates = df[
    df.duplicated(subset=["normalized_text"], keep=False)
].sort_values("normalized_text")

print(
    duplicates[
        [
            "text",
            "primary_department",
            "secondary_department",
            "complaint_type",
            "language"
        ]
    ].to_string(index=False)
)

print("\n3. Duplicate texts with conflicting primary departments:")
for text, group in duplicates.groupby("normalized_text"):
    departments = group["primary_department"].unique()

    if len(departments) > 1:
        print("\nComplaint:", text)
        print("Departments:", departments)

print("\nDuplicate audit completed.")
