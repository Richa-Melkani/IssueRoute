
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "research_dataset_clean.csv"

df = pd.read_csv(DATASET_PATH)
df = df.dropna(subset=["text", "primary_department"])
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]

X = df["text"]
y = df["primary_department"]

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

split_ratios = {
    "75:25": 0.25,
    "80:20": 0.20,
    "90:10": 0.10
}

results = []

for split_name, test_size in split_ratios.items():

    print(f"\n========== {split_name} SPLIT ==========")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    for model_name, classifier in models.items():

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(
            y_test, predictions,
            average="macro",
            zero_division=0
        )

        results.append({
            "Split": split_name,
            "Model": model_name,
            "Accuracy": accuracy,
            "Macro F1": macro_f1,
            "Training Samples": len(X_train),
            "Testing Samples": len(X_test)
        })

        print(
            model_name,
            "| Accuracy:", round(accuracy, 4),
            "| Macro F1:", round(macro_f1, 4)
        )

results_df = pd.DataFrame(results)

output_path = (
    BASE_DIR / "model_analysis" / "split_ratio_results.csv"
)

results_df.to_csv(output_path, index=False)

print("\n========== FINAL SPLIT COMPARISON ==========")
print(results_df.to_string(index=False))
print("\nResults saved to:", output_path)
