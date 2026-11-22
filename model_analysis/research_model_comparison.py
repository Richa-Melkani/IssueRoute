
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# 1. Load cleaned research dataset
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "research_dataset_clean.csv"

df = pd.read_csv(DATASET_PATH)
df = df.dropna(subset=["text", "primary_department"])

df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]

X = df["text"]
y = df["primary_department"]

# 2. Create one common stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Define six algorithms
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = []

print("\n========== SIX-MODEL COMPARISON ==========")
print("Total records:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 4. Train and evaluate each model
for name, classifier in models.items():

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": macro_f1
    })

    print(f"\n{name}")
    print("Accuracy:", round(accuracy, 4))
    print("Macro Precision:", round(precision, 4))
    print("Macro Recall:", round(recall, 4))
    print("Macro F1:", round(macro_f1, 4))

# 5. Display comparison
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Macro F1", ascending=False
)

print("\n========== FINAL COMPARISON TABLE ==========")
print(results_df.to_string(index=False))

# 6. Save results
OUTPUT_PATH = BASE_DIR / "model_analysis" / "research_model_comparison.csv"
results_df.to_csv(OUTPUT_PATH, index=False)

print("\nResults saved to:", OUTPUT_PATH)
print("\nComparison completed.")
