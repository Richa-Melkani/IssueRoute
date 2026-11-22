import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

data = pd.read_csv(
    BASE_DIR / "model_analysis" / "cv_predictions.csv"
)

wrong = data[data["correct"] == False].copy()

print("Total complaints:", len(data))
print("Incorrect predictions:", len(wrong))
print("\nMost common classification mistakes:")

mistakes = (
    wrong.groupby(["actual", "predicted"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print(mistakes.to_string(index=False))

print("\nMisclassified complaints:")

for _, row in wrong.iterrows():
    print("\nActual:", row["actual"])
    print("Predicted:", row["predicted"])
    print("Complaint:", row["text"])