
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "research_dataset_clean.csv"

df = pd.read_csv(DATASET_PATH)

# Remove empty complaint text
df = df.dropna(subset=["text", "primary_department"])
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]

X = df["text"]
y = df["primary_department"]

# Stratified 80:20 split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== DATASET SPLIT CHECK ==========")
print("Total records:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining department distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting department distribution:")
print(y_test.value_counts().sort_index())

# Check whether any complaint text appears in both sets
overlap = set(X_train.str.strip().str.lower()) & set(
    X_test.str.strip().str.lower()
)

print("\nTrain-test text overlap:", len(overlap))

print("\nSplit check completed.")
