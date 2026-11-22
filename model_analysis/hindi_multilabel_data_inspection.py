
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    BASE_DIR / "dataset" / "research_dataset_clean.csv"
)

hindi_multi = df[
    (df["language"] == "Hindi") &
    (df["complaint_type"] == "multi")
].copy()

departments = [
    "Administration",
    "Fees / Finance",
    "Maintenance"
]

for department in departments:

    selected = hindi_multi[
        (hindi_multi["primary_department"] == department) |
        (hindi_multi["secondary_department"] == department)
    ]

    print("\n" + "=" * 65)
    print("DEPARTMENT:", department)
    print("Number of Hindi multi complaints:", len(selected))
    print("=" * 65)

    for _, row in selected.iterrows():

        print("\nComplaint:", row["text"])
        print("Primary:", row["primary_department"])
        print("Secondary:", row["secondary_department"])

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    BASE_DIR / "dataset" / "research_dataset_clean.csv"
)

multi = df[df["complaint_type"] == "multi"]

print("\nPRIMARY DEPARTMENT COUNTS")
print(multi["primary_department"].value_counts())

print("\nSECONDARY DEPARTMENT COUNTS")
print(multi["secondary_department"].value_counts())

print("\nHINDI PRIMARY DEPARTMENT COUNTS")
print(
    multi[multi["language"] == "Hindi"]
    ["primary_department"]
    .value_counts()
)

print("\nHINDI SECONDARY DEPARTMENT COUNTS")
print(
    multi[multi["language"] == "Hindi"]
    ["secondary_department"]
    .value_counts()
)

print("\nInspection completed.")
