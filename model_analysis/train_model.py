import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    hamming_loss
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

BASELINE_PATH = DATASET_DIR / "baseline_dataset.csv"
RESEARCH_PATH = DATASET_DIR / "research_dataset.csv"
MODEL_PATH = MODEL_DIR / "issue_route_model.pkl"


# ============================================================
# 1. BASELINE DATASET
# ============================================================

print("\n" + "=" * 60)
print("1. BASELINE DATASET")
print("=" * 60)

df = pd.read_csv(BASELINE_PATH)

print(df.head())
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

X = df["text"]
y = df["department"]

X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train_base))
print("Testing samples:", len(X_test_base))


# ============================================================
# 2. BASELINE TF-IDF + LINEAR SVM
# ============================================================

print("\n" + "=" * 60)
print("2. BASELINE TF-IDF + LINEAR SVM")
print("=" * 60)

tfidf_base = TfidfVectorizer()

X_train_tfidf = tfidf_base.fit_transform(X_train_base)
X_test_tfidf = tfidf_base.transform(X_test_base)

print("TF-IDF train shape:", X_train_tfidf.shape)
print("TF-IDF test shape:", X_test_tfidf.shape)

model_base = LinearSVC()
model_base.fit(X_train_tfidf, y_train_base)

y_pred_base = model_base.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test_base, y_pred_base) * 100)
print("Macro F1:", f1_score(y_test_base, y_pred_base, average="macro"))

print(classification_report(y_test_base, y_pred_base, zero_division=0))


# ============================================================
# 3. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("3. MODEL COMPARISON")
print("=" * 60)

comparison_models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC()
}

comparison_results = []

for name, clf in comparison_models.items():
    clf.fit(X_train_tfidf, y_train_base)
    pred = clf.predict(X_test_tfidf)

    comparison_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test_base, pred),
        "Macro F1": f1_score(y_test_base, pred, average="macro")
    })

print(pd.DataFrame(comparison_results))


# ============================================================
# 4. BIGRAM TF-IDF + LINEAR SVM
# ============================================================

print("\n" + "=" * 60)
print("4. BIGRAM TF-IDF + LINEAR SVM")
print("=" * 60)

tfidf_bigram = TfidfVectorizer(ngram_range=(1, 2))

X_train_bigram = tfidf_bigram.fit_transform(X_train_base)
X_test_bigram = tfidf_bigram.transform(X_test_base)

model_bigram = LinearSVC()
model_bigram.fit(X_train_bigram, y_train_base)

y_pred_bigram = model_bigram.predict(X_test_bigram)

print("Accuracy:", accuracy_score(y_test_base, y_pred_bigram) * 100)
print("Macro F1:", f1_score(y_test_base, y_pred_bigram, average="macro"))


# ============================================================
# 5. CHARACTER TF-IDF + LINEAR SVM
# ============================================================

print("\n" + "=" * 60)
print("5. CHARACTER TF-IDF + LINEAR SVM")
print("=" * 60)

tfidf_char_base = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5)
)

X_train_char_base = tfidf_char_base.fit_transform(X_train_base)
X_test_char_base = tfidf_char_base.transform(X_test_base)

model_char_base = LinearSVC(C=1.0)
model_char_base.fit(X_train_char_base, y_train_base)

y_pred_char_base = model_char_base.predict(X_test_char_base)

print("Accuracy:", accuracy_score(y_test_base, y_pred_char_base) * 100)
print("Macro F1:", f1_score(y_test_base, y_pred_char_base, average="macro"))


# ============================================================
# 6. BALANCED SVM
# ============================================================

print("\n" + "=" * 60)
print("6. BALANCED SVM")
print("=" * 60)

model_balanced = LinearSVC(class_weight="balanced")
model_balanced.fit(X_train_tfidf, y_train_base)

y_pred_balanced = model_balanced.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test_base, y_pred_balanced) * 100)
print("Macro F1:", f1_score(y_test_base, y_pred_balanced, average="macro"))


# ============================================================
# 7. RESEARCH DATASET
# ============================================================

print("\n" + "=" * 60)
print("7. RESEARCH DATASET")
print("=" * 60)

research_df = pd.read_csv(RESEARCH_PATH)

# IMPORTANT:
# Remove exact duplicate rows BEFORE any research experiment.
# This prevents duplicate records from affecting train/test results.
before_duplicates = len(research_df)

research_df = research_df.drop_duplicates(
    subset=["text", "primary_department", "secondary_department",
            "complaint_type", "language"]
).reset_index(drop=True)

removed_duplicates = before_duplicates - len(research_df)

print("Rows before duplicate removal:", before_duplicates)
print("Duplicate rows removed:", removed_duplicates)
print("Rows after duplicate removal:", len(research_df))
print("Shape:", research_df.shape)

print("\nComplaint types:")
print(research_df["complaint_type"].value_counts())

print("\nLanguages:")
print(research_df["language"].value_counts())

print("\nPrimary departments:")
print(research_df["primary_department"].value_counts())

print("\nDuplicate rows after cleaning:",
      research_df.duplicated(
          subset=["text", "primary_department", "secondary_department",
                  "complaint_type", "language"]
      ).sum())


# ============================================================
# COMMON RESEARCH SPLIT
# ============================================================

X_research = research_df["text"]


# ============================================================
# 8. EXPERIMENT 1 - COMPLAINT TYPE
# ============================================================

print("\n" + "=" * 60)
print("8. EXPERIMENT 1 - COMPLAINT TYPE")
print("=" * 60)

y_type = research_df["complaint_type"]

X_train_type, X_test_type, y_train_type, y_test_type = train_test_split(
    X_research,
    y_type,
    test_size=0.20,
    random_state=42,
    stratify=y_type
)

tfidf_type = TfidfVectorizer()

X_train_type_tfidf = tfidf_type.fit_transform(X_train_type)
X_test_type_tfidf = tfidf_type.transform(X_test_type)

model_type = LinearSVC()
model_type.fit(X_train_type_tfidf, y_train_type)

y_pred_type = model_type.predict(X_test_type_tfidf)

print("Accuracy:", accuracy_score(y_test_type, y_pred_type) * 100)
print("Macro F1:", f1_score(y_test_type, y_pred_type, average="macro"))
print(classification_report(y_test_type, y_pred_type, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(y_test_type, y_pred_type))


# ============================================================
# 9. EXPERIMENT 2 - MULTI VS NOT-MULTI
# ============================================================

print("\n" + "=" * 60)
print("9. EXPERIMENT 2 - MULTI VS NOT-MULTI")
print("=" * 60)

y_binary = research_df["complaint_type"].apply(
    lambda x: "multi" if x == "multi" else "not_multi"
)

X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(
    X_research,
    y_binary,
    test_size=0.20,
    random_state=42,
    stratify=y_binary
)

tfidf_binary = TfidfVectorizer()

X_train_binary_tfidf = tfidf_binary.fit_transform(X_train_binary)
X_test_binary_tfidf = tfidf_binary.transform(X_test_binary)

model_binary = LinearSVC()
model_binary.fit(X_train_binary_tfidf, y_train_binary)

y_pred_binary = model_binary.predict(X_test_binary_tfidf)

print("Accuracy:", accuracy_score(y_test_binary, y_pred_binary) * 100)
print("Macro F1:", f1_score(y_test_binary, y_pred_binary, average="macro"))
print(classification_report(y_test_binary, y_pred_binary, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(y_test_binary, y_pred_binary))


# ============================================================
# 10. EXPERIMENT 3 - PRIMARY DEPARTMENT CLASSIFICATION
# ============================================================

print("\n" + "=" * 60)
print("10. EXPERIMENT 3 - PRIMARY DEPARTMENT CLASSIFICATION")
print("=" * 60)

y_department = research_df["primary_department"]

X_train_dept, X_test_dept, y_train_dept, y_test_dept = train_test_split(
    X_research,
    y_department,
    test_size=0.20,
    random_state=42,
    stratify=y_department
)

tfidf_dept = TfidfVectorizer()

X_train_dept_tfidf = tfidf_dept.fit_transform(X_train_dept)
X_test_dept_tfidf = tfidf_dept.transform(X_test_dept)

model_dept = LinearSVC()
model_dept.fit(X_train_dept_tfidf, y_train_dept)

y_pred_dept = model_dept.predict(X_test_dept_tfidf)

dept_accuracy = accuracy_score(y_test_dept, y_pred_dept)
dept_macro_f1 = f1_score(
    y_test_dept,
    y_pred_dept,
    average="macro"
)

print("Accuracy:", dept_accuracy * 100)
print("Macro F1:", dept_macro_f1)

print(classification_report(
    y_test_dept,
    y_pred_dept,
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(y_test_dept, y_pred_dept))


# ============================================================
# 11. EXPERIMENT 4 - UNKNOWN VS KNOWN
# ============================================================

print("\n" + "=" * 60)
print("11. EXPERIMENT 4 - UNKNOWN VS KNOWN")
print("=" * 60)

y_unknown = research_df["primary_department"].apply(
    lambda x: "unknown" if x == "Unknown" else "known"
)

X_train_unknown, X_test_unknown, y_train_unknown, y_test_unknown = train_test_split(
    X_research,
    y_unknown,
    test_size=0.20,
    random_state=42,
    stratify=y_unknown
)

tfidf_unknown = TfidfVectorizer()

X_train_unknown_tfidf = tfidf_unknown.fit_transform(X_train_unknown)
X_test_unknown_tfidf = tfidf_unknown.transform(X_test_unknown)

model_unknown = LinearSVC()
model_unknown.fit(X_train_unknown_tfidf, y_train_unknown)

y_pred_unknown = model_unknown.predict(X_test_unknown_tfidf)

print("Accuracy:", accuracy_score(y_test_unknown, y_pred_unknown) * 100)
print("Macro F1:", f1_score(
    y_test_unknown,
    y_pred_unknown,
    average="macro"
))

print(classification_report(
    y_test_unknown,
    y_pred_unknown,
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(y_test_unknown, y_pred_unknown))


# ============================================================
# 12. EXPERIMENT 5 - LANGUAGE-WISE DEPARTMENT PERFORMANCE
# ============================================================

print("\n" + "=" * 60)
print("12. EXPERIMENT 5 - LANGUAGE-WISE DEPARTMENT PERFORMANCE")
print("=" * 60)

language_results = []

for language in ["English", "Hindi", "Hinglish"]:
    mask = X_test_dept.index.isin(
        research_df.index[research_df["language"] == language]
    )

    language_true = y_test_dept[mask]
    language_pred = pd.Series(
        y_pred_dept,
        index=y_test_dept.index
    )[mask]

    if len(language_true) > 0:
        language_results.append({
            "Language": language,
            "Accuracy": accuracy_score(
                language_true,
                language_pred
            ),
            "Samples": len(language_true)
        })

language_results_df = pd.DataFrame(language_results)

print(language_results_df)

error_mask = y_test_dept.values != y_pred_dept

language_error_df = pd.DataFrame({
    "text": X_test_dept.values,
    "actual": y_test_dept.values,
    "predicted": y_pred_dept,
    "language": research_df.loc[
        X_test_dept.index, "language"
    ].values
})

print("\nLanguage-wise errors:")
print(
    language_error_df[
        language_error_df["actual"] !=
        language_error_df["predicted"]
    ][["text", "actual", "predicted", "language"]]
)


# ============================================================
# 13. EXPERIMENT 6 - SECONDARY DEPARTMENT CLASSIFICATION
# ============================================================

print("\n" + "=" * 60)
print("13. EXPERIMENT 6 - SECONDARY DEPARTMENT CLASSIFICATION")
print("=" * 60)

multi_df = research_df[
    research_df["complaint_type"] == "multi"
].copy()

X_secondary = multi_df["text"]
y_secondary = multi_df["secondary_department"]

X_train_secondary, X_test_secondary, y_train_secondary, y_test_secondary = train_test_split(
    X_secondary,
    y_secondary,
    test_size=0.20,
    random_state=42,
    stratify=y_secondary
)

tfidf_secondary = TfidfVectorizer()

X_train_secondary_tfidf = tfidf_secondary.fit_transform(
    X_train_secondary
)

X_test_secondary_tfidf = tfidf_secondary.transform(
    X_test_secondary
)

model_secondary = LinearSVC()
model_secondary.fit(
    X_train_secondary_tfidf,
    y_train_secondary
)

y_pred_secondary = model_secondary.predict(
    X_test_secondary_tfidf
)

print("Accuracy:",
      accuracy_score(
          y_test_secondary,
          y_pred_secondary
      ) * 100)

print("Macro F1:",
      f1_score(
          y_test_secondary,
          y_pred_secondary,
          average="macro"
      ))

print(classification_report(
    y_test_secondary,
    y_pred_secondary,
    zero_division=0
))

print("Confusion matrix:")
print(
    confusion_matrix(
        y_test_secondary,
        y_pred_secondary
    )
)


# ============================================================
# 14. EXPERIMENT 7 - UNCERTAINTY-AWARE ROUTING
# ============================================================

print("\n" + "=" * 60)
print("14. EXPERIMENT 7 - UNCERTAINTY-AWARE ROUTING")
print("=" * 60)

decision_scores = model_dept.decision_function(
    X_test_dept_tfidf
)

sorted_scores = np.sort(
    decision_scores,
    axis=1
)

margin = (
    sorted_scores[:, -1] -
    sorted_scores[:, -2]
)

print("Decision score shape:", decision_scores.shape)
print("Margin shape:", margin.shape)
print("Minimum margin:", margin.min())
print("Maximum margin:", margin.max())
print("Average margin:", margin.mean())

threshold = 0.3

review_mask = margin < threshold

reviewed = review_mask.sum()
total_errors = (y_test_dept.values != y_pred_dept).sum()

error_mask_dept = (
    y_test_dept.values != y_pred_dept
)

errors_caught = (
    review_mask & error_mask_dept
).sum()

simulated_correct = (
    (~review_mask & ~error_mask_dept).sum()
    + review_mask.sum()
)

simulated_accuracy = (
    simulated_correct /
    len(y_test_dept)
)

simulated_true = y_test_dept.copy()
simulated_pred = y_pred_dept.copy()

simulated_pred[review_mask] = (
    simulated_true.values[review_mask]
)

simulated_macro_f1 = f1_score(
    simulated_true,
    simulated_pred,
    average="macro"
)

print("Threshold:", threshold)
print("Reviewed:", reviewed, "/", len(y_test_dept))
print("Errors caught:", errors_caught, "/", total_errors)
print(
    "Simulated accuracy after perfect human correction:",
    simulated_accuracy * 100
)
print("Simulated Macro F1:", simulated_macro_f1)


# ============================================================
# 15. EXPERIMENT 8 - CHARACTER TF-IDF ABLATION
# ============================================================

print("\n" + "=" * 60)
print("15. EXPERIMENT 8 - CHARACTER TF-IDF ABLATION")
print("=" * 60)

tfidf_char = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5)
)

X_train_char = tfidf_char.fit_transform(X_train_dept)
X_test_char = tfidf_char.transform(X_test_dept)

model_char = LinearSVC(C=1.0)

model_char.fit(
    X_train_char,
    y_train_dept
)

y_pred_char = model_char.predict(
    X_test_char
)

print("Accuracy:",
      accuracy_score(
          y_test_dept,
          y_pred_char
      ) * 100)

print("Macro F1:",
      f1_score(
          y_test_dept,
          y_pred_char,
          average="macro"
      ))

print(classification_report(
    y_test_dept,
    y_pred_char,
    zero_division=0
))

print("Confusion matrix:")
print(
    confusion_matrix(
        y_test_dept,
        y_pred_char
    )
)


# ============================================================
# 16. EXPERIMENT 9 - THRESHOLD SENSITIVITY
# ============================================================

print("\n" + "=" * 60)
print("16. EXPERIMENT 9 - THRESHOLD SENSITIVITY")
print("=" * 60)

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

threshold_results = []

for current_threshold in thresholds:

    review_mask_temp = (
        margin < current_threshold
    )

    simulated_pred_temp = y_pred_dept.copy()

    simulated_pred_temp[review_mask_temp] = (
        y_test_dept.values[review_mask_temp]
    )

    threshold_results.append({
        "Threshold": current_threshold,
        "Reviewed": int(review_mask_temp.sum()),
        "Simulated Accuracy":
            accuracy_score(
                y_test_dept,
                simulated_pred_temp
            ),
        "Simulated Macro F1":
            f1_score(
                y_test_dept,
                simulated_pred_temp,
                average="macro"
            )
    })

threshold_results_df = pd.DataFrame(
    threshold_results
)

threshold_results_df["Review %"] = (
    threshold_results_df["Reviewed"] /
    len(y_test_dept) * 100
)

print(threshold_results_df)

best_threshold_row = threshold_results_df.loc[
    threshold_results_df["Simulated Macro F1"].idxmax()
]

print("\nBest tested threshold by Macro F1:")
print(best_threshold_row)


# ============================================================
# 17. EXPERIMENT 10 - EXPLAINABILITY
# ============================================================

print("\n" + "=" * 60)
print("17. EXPERIMENT 10 - EXPLAINABILITY")
print("=" * 60)

feature_names = tfidf_dept.get_feature_names_out()
coefficients = model_dept.coef_
classes = model_dept.classes_

print("Number of TF-IDF features:", len(feature_names))
print("Number of classes:", len(classes))
print("Classes:", classes)

for class_index, class_name in enumerate(classes):

    top_indices = np.argsort(
        coefficients[class_index]
    )[-10:][::-1]

    top_features = feature_names[top_indices]

    print(f"\n{class_name}:")
    print(list(top_features))


def explain_prediction(text, model, vectorizer, top_n=5):

    vector = vectorizer.transform([text])

    predicted = model.predict(vector)[0]

    class_index = list(
        model.classes_
    ).index(predicted)

    feature_names_local = (
        vectorizer.get_feature_names_out()
    )

    feature_values = vector.toarray()[0]
    class_coefficients = (
        model.coef_[class_index]
    )

    contributions = (
        feature_values *
        class_coefficients
    )

    top_indices = np.argsort(
        contributions
    )[-top_n:][::-1]

    print("\nComplaint:", text)
    print("Predicted department:", predicted)
    print("Top contributing features:")

    for index in top_indices:
        if contributions[index] > 0:
            print(
                f"  {feature_names_local[index]}: "
                f"{contributions[index]:.4f}"
            )


explain_prediction(
    "How can I request a correction in my academic record?",
    model_dept,
    tfidf_dept
)

explain_prediction(
    "My university portal is not allowing me to login.",
    model_dept,
    tfidf_dept
)

explain_prediction(
    "There is a water leakage in my hostel bathroom.",
    model_dept,
    tfidf_dept
)

explain_prediction(
    "There is water leakage in the bathroom.",
    model_dept,
    tfidf_dept
)


# ============================================================
# 18. EXPERIMENT 11 - LANGUAGE-WISE CHAR VS WORD TF-IDF
# ============================================================

print("\n" + "=" * 60)
print("18. EXPERIMENT 11 - LANGUAGE-WISE CHAR VS WORD TF-IDF")
print("=" * 60)

language_char_results = []

char_predictions_series = pd.Series(
    y_pred_char,
    index=y_test_dept.index
)

word_predictions_series = pd.Series(
    y_pred_dept,
    index=y_test_dept.index
)

for language in ["English", "Hindi", "Hinglish"]:

    language_mask = (
        research_df.loc[
            y_test_dept.index,
            "language"
        ] == language
    )

    true_values = y_test_dept[language_mask]
    word_values = word_predictions_series[language_mask]
    char_values = char_predictions_series[language_mask]

    word_accuracy = accuracy_score(
        true_values,
        word_values
    )

    char_accuracy = accuracy_score(
        true_values,
        char_values
    )

    language_char_results.append({
        "Language": language,
        "Word TF-IDF Accuracy": word_accuracy,
        "Char TF-IDF Accuracy": char_accuracy,
        "Difference (pp)":
            (char_accuracy - word_accuracy) * 100,
        "Samples": len(true_values)
    })

print(
    pd.DataFrame(language_char_results)
)


# ============================================================
# 19. EXPERIMENT 12 - MULTI-DEPARTMENT ROUTING
# ============================================================

print("\n" + "=" * 60)
print("19. EXPERIMENT 12 - MULTI-DEPARTMENT ROUTING")
print("=" * 60)

multi_df = research_df[
    research_df["complaint_type"] == "multi"
].copy()

print(
    "Multi-issue complaints:",
    len(multi_df)
)

multi_df["target_departments"] = (
    multi_df["primary_department"]
    + " + "
    + multi_df["secondary_department"]
)

print("\nTarget department combinations:")
print(
    multi_df["target_departments"].value_counts()
)

multi_df["department_pair"] = multi_df.apply(
    lambda row: " + ".join(
        sorted([
            row["primary_department"],
            row["secondary_department"]
        ])
    ),
    axis=1
)

print("\nUnique unordered pairs:",
      multi_df["department_pair"].nunique())

print(
    multi_df["department_pair"].value_counts()
)

departments = [
    "Academics",
    "Administration",
    "Fees / Finance",
    "Hostels",
    "IT Support",
    "Maintenance",
    "Security / Discipline",
    "Transport"
]

for department in departments:
    multi_df[department] = (
        (multi_df["primary_department"] == department) |
        (multi_df["secondary_department"] == department)
    ).astype(int)

X_multi = multi_df["text"]
Y_multi = multi_df[departments]

X_train_multi, X_test_multi, Y_train_multi, Y_test_multi = train_test_split(
    X_multi,
    Y_multi,
    test_size=0.20,
    random_state=42
)

print("Train shape before TF-IDF:", X_train_multi.shape)
print("Test shape before TF-IDF:", X_test_multi.shape)

tfidf_multi = TfidfVectorizer()

X_train_multi_tfidf = (
    tfidf_multi.fit_transform(X_train_multi)
)

X_test_multi_tfidf = (
    tfidf_multi.transform(X_test_multi)
)

print("Train shape:", X_train_multi_tfidf.shape)
print("Test shape:", X_test_multi_tfidf.shape)

multi_model = OneVsRestClassifier(
    LinearSVC()
)

multi_model.fit(
    X_train_multi_tfidf,
    Y_train_multi
)

Y_pred_multi = multi_model.predict(
    X_test_multi_tfidf
)

print(
    "Subset accuracy:",
    accuracy_score(
        Y_test_multi,
        Y_pred_multi
    ) * 100
)

print(
    "Hamming loss:",
    hamming_loss(
        Y_test_multi,
        Y_pred_multi
    )
)

print(
    "Macro F1:",
    f1_score(
        Y_test_multi,
        Y_pred_multi,
        average="macro",
        zero_division=0
    )
)

print(
    classification_report(
        Y_test_multi,
        Y_pred_multi,
        target_names=departments,
        zero_division=0
    )
)

false_positive_count = (
    (Y_pred_multi == 1) &
    (Y_test_multi.values == 0)
).sum()

false_negative_count = (
    (Y_pred_multi == 0) &
    (Y_test_multi.values == 1)
).sum()

print("False positives:", false_positive_count)
print("False negatives:", false_negative_count)


# ------------------------------------------------------------
# Multi-label decision scores
# ------------------------------------------------------------

decision_multi = multi_model.decision_function(
    X_test_multi_tfidf
)

print(
    "Decision score shape:",
    decision_multi.shape
)

print(
    "Minimum score:",
    decision_multi.min()
)

print(
    "Maximum score:",
    decision_multi.max()
)


# ------------------------------------------------------------
# Multi-label threshold sensitivity
# ------------------------------------------------------------

multi_thresholds = [
    -0.5,
    -0.4,
    -0.3,
    -0.2,
    -0.1,
    0.0
]

multi_threshold_results = []

for current_threshold in multi_thresholds:

    Y_pred_threshold = (
        decision_multi >= current_threshold
    ).astype(int)

    multi_threshold_results.append({
        "Threshold": current_threshold,
        "Subset Accuracy":
            accuracy_score(
                Y_test_multi,
                Y_pred_threshold
            ),
        "Hamming Loss":
            hamming_loss(
                Y_test_multi,
                Y_pred_threshold
            ),
        "Macro F1":
            f1_score(
                Y_test_multi,
                Y_pred_threshold,
                average="macro",
                zero_division=0
            )
    })

multi_threshold_df = pd.DataFrame(
    multi_threshold_results
)

print("\nMulti-label threshold sensitivity:")
print(multi_threshold_df)

best_multi_threshold = multi_threshold_df.loc[
    multi_threshold_df["Macro F1"].idxmax()
]

print("\nBest tested threshold by Macro F1:")
print(best_multi_threshold)

best_threshold = best_multi_threshold["Threshold"]

Y_pred_best_multi = (
    decision_multi >= best_threshold
).astype(int)

print(
    f"\nAt threshold {best_threshold}"
)

best_false_positive_count = (
    (Y_pred_best_multi == 1) &
    (Y_test_multi.values == 0)
).sum()

best_false_negative_count = (
    (Y_pred_best_multi == 0) &
    (Y_test_multi.values == 1)
).sum()

print(
    "False positives:",
    best_false_positive_count
)

print(
    "False negatives:",
    best_false_negative_count
)

print(
    "Subset accuracy:",
    accuracy_score(
        Y_test_multi,
        Y_pred_best_multi
    ) * 100
)

print(
    "Hamming loss:",
    hamming_loss(
        Y_test_multi,
        Y_pred_best_multi
    )
)

print(
    "Macro F1:",
    f1_score(
        Y_test_multi,
        Y_pred_best_multi,
        average="macro",
        zero_division=0
    )
)


# ============================================================
# 20. FINAL MODEL SAVE
# ============================================================

print("\n" + "=" * 60)
print("20. FINAL MODEL SAVE")
print("=" * 60)

# IMPORTANT:
# Save the vectorizer + classifier together.
# Flask can then load one file and call:
# pipeline.predict(["complaint text"])

final_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("svm", LinearSVC())
])

final_model.fit(
    X_train_dept,
    y_train_dept
)

joblib.dump(
    final_model,
    MODEL_PATH
)

print("Model saved successfully:")
print(MODEL_PATH)

# Verify saved model
loaded_model = joblib.load(MODEL_PATH)

print("Saved model loaded successfully.")

demo_complaints = [
    "My hostel room fan is not working.",
    "My university email account is not working.",
    "My semester fee payment is failing.",
    "The college bus is late every morning."
]

print("\nDemo predictions:")

for complaint in demo_complaints:
    prediction = loaded_model.predict([complaint])[0]
    print(f"{complaint} -> {prediction}")


# ============================================================
# END
# ============================================================

print("\n" + "=" * 60)
print("ALL EXPERIMENTS COMPLETED")
print("=" * 60)
