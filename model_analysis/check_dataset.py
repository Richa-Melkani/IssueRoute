
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

datasets = {
    "Baseline Dataset": {
        "path": BASE_DIR / "dataset" / "baseline_dataset.csv",
        "label": "department"
    },
    "Research Dataset": {
        "path": BASE_DIR / "dataset" / "research_dataset.csv",
        "label": "primary_department"
    }
}

for name, info in datasets.items():
    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    df = pd.read_csv(info["path"])

    print("\n1. Dataset shape:")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\n2. Column names:")
    print(df.columns.tolist())

    print("\n3. Missing values:")
    print(df.isnull().sum())

    print("\n4. Exact duplicate rows:")
    print(df.duplicated().sum())

    print("\n5. Duplicate complaint texts:")
    print(df["text"].duplicated().sum())

    print("\n6. Department distribution:")
    print(df[info["label"]].value_counts(dropna=False))

    print("\n7. Department percentages:")
    print(
        (df[info["label"]].value_counts(normalize=True, dropna=False) * 100)
        .round(2)
    )

print("\nDataset checking completed.")
