import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from scipy.sparse import hstack

from sklearn.metrics import f1_score



# =========================
# 1. Load Dataset
# =========================

data_path = "../dataset/complaints.csv"

df = pd.read_csv(data_path)

print(df.head())

print("Dataset shape:", df.shape)

print("Columns:", df.columns.tolist())


# =========================
# 2. Separate Input and Target
# =========================

X = df["text"]

y = df["department"]

print("Input (X):")

print(X.head())

print("Target (y):")

print(y.head())


# =========================
# 3. Train-Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)

print("Testing data:", X_test.shape)


# =========================
# 4. Normal TF-IDF
# =========================

tfidf = TfidfVectorizer()

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)

print("Training TF-IDF shape:", X_train_tfidf.shape)

print("Testing TF-IDF shape:", X_test_tfidf.shape)


# =========================
# 5. Linear SVM
# =========================

model = LinearSVC()

model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("First 10 predictions:")

print(y_pred[:10])


# =========================
# 6. Normal TF-IDF Accuracy
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("Normal TF-IDF Accuracy:", accuracy)


# =========================
# 7. Normal TF-IDF Classification Report
# =========================

print(classification_report(y_test, y_pred))


# =========================
# 8. Normal TF-IDF Confusion Matrix
# =========================

cm = confusion_matrix(y_test, y_pred)

print("Normal TF-IDF Confusion Matrix:")

print(cm)


# ==================================================
# 9. Bigram TF-IDF
# ==================================================

tfidf_bigram = TfidfVectorizer(ngram_range=(1, 2))

X_train_bigram = tfidf_bigram.fit_transform(X_train)

X_test_bigram = tfidf_bigram.transform(X_test)

print("Bigram Training shape:", X_train_bigram.shape)

print("Bigram Testing shape:", X_test_bigram.shape)


# =========================
# 10. Linear SVM with Bigram
# =========================

model_bigram = LinearSVC()

model_bigram.fit(X_train_bigram, y_train)

y_pred_bigram = model_bigram.predict(X_test_bigram)


# =========================
# 11. Bigram Accuracy
# =========================

accuracy_bigram = accuracy_score(y_test, y_pred_bigram)

print("Bigram Accuracy:", accuracy_bigram)


# =========================
# 12. Bigram Classification Report
# =========================

print(classification_report(y_test, y_pred_bigram))


# =========================
# 13. Bigram Confusion Matrix
# =========================

cm_bigram = confusion_matrix(y_test, y_pred_bigram)

print("Bigram Confusion Matrix:")

print(cm_bigram)

tfidf_char = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5)
)

X_train_char = tfidf_char.fit_transform(X_train)
X_test_char = tfidf_char.transform(X_test)

print("Character TF-IDF Training shape:", X_train_char.shape)
print("Character TF-IDF Testing shape:", X_test_char.shape)

model_char = LinearSVC()

model_char.fit(X_train_char, y_train)

y_pred_char = model_char.predict(X_test_char)

accuracy_char = accuracy_score(y_test, y_pred_char)

print("Character TF-IDF Accuracy:", accuracy_char)

print(classification_report(y_test, y_pred_char))


cm_char = confusion_matrix(y_test, y_pred_char)

print("Character TF-IDF Confusion Matrix:")
print(cm_char)


model_balanced = LinearSVC(class_weight="balanced")

model_balanced.fit(X_train_tfidf, y_train)

y_pred_balanced = model_balanced.predict(X_test_tfidf)

accuracy_balanced = accuracy_score(y_test, y_pred_balanced)

print("Balanced SVM Accuracy:", accuracy_balanced)

from sklearn.metrics import f1_score

macro_f1_balanced = f1_score(y_test, y_pred_balanced, average="macro")

print("Balanced SVM Macro F1:", macro_f1_balanced)

X_train_combined = hstack([X_train_tfidf, X_train_char])
X_test_combined = hstack([X_test_tfidf, X_test_char])

print("Combined Training shape:", X_train_combined.shape)
print("Combined Testing shape:", X_test_combined.shape)


model_combined = LinearSVC()

model_combined.fit(X_train_combined, y_train)

y_pred_combined = model_combined.predict(X_test_combined)

accuracy_combined = accuracy_score(y_test, y_pred_combined)

print("Combined TF-IDF Accuracy:", accuracy_combined)



macro_f1_combined = f1_score(
    y_test,
    y_pred_combined,
    average="macro"
)

print("Combined TF-IDF Macro F1:", macro_f1_combined)


model_char_c05 = LinearSVC(C=0.5)

model_char_c05.fit(X_train_char, y_train)

y_pred_char_c05 = model_char_c05.predict(X_test_char)

accuracy_char_c05 = accuracy_score(y_test, y_pred_char_c05)

print("Character TF-IDF + SVM (C=0.5) Accuracy:", accuracy_char_c05)

model_char_c2 = LinearSVC(C=2.0)

model_char_c2.fit(X_train_char, y_train)

y_pred_char_c2 = model_char_c2.predict(X_test_char)

accuracy_char_c2 = accuracy_score(y_test, y_pred_char_c2)

print("Character TF-IDF + SVM (C=2.0) Accuracy:", accuracy_char_c2)


model_char_c01 = LinearSVC(C=0.1)

model_char_c01.fit(X_train_char, y_train)

y_pred_char_c01 = model_char_c01.predict(X_test_char)

accuracy_char_c01 = accuracy_score(y_test, y_pred_char_c01)

print("Character TF-IDF + SVM (C=0.1) Accuracy:", accuracy_char_c01)

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred_char
})

wrong_predictions = results[
    results["Actual"] != results["Predicted"]
]

print("Total wrong predictions:", len(wrong_predictions))
print(wrong_predictions.head(20))

wrong_indices = y_test.index[y_test != y_pred_char]

wrong_details = df.loc[wrong_indices, ["text", "department"]].copy()
wrong_details["Predicted"] = y_pred_char[
    [list(y_test.index).index(i) for i in wrong_indices]
]

print(wrong_details.head(20).to_string(index=False))

error_pairs = (
    wrong_predictions
    .groupby(["Actual", "Predicted"])
    .size()
    .sort_values(ascending=False)
)

print(error_pairs)

from sklearn.linear_model import LogisticRegression

model_char_lr = LogisticRegression(max_iter=1000)

model_char_lr.fit(X_train_char, y_train)

y_pred_char_lr = model_char_lr.predict(X_test_char)

accuracy_char_lr = accuracy_score(y_test, y_pred_char_lr)

print("Character TF-IDF + Logistic Regression Accuracy:", accuracy_char_lr)


macro_f1_char_lr = f1_score(
    y_test,
    y_pred_char_lr,
    average="macro"
)

print("Character TF-IDF + Logistic Regression Macro F1:", macro_f1_char_lr)


from sklearn.naive_bayes import MultinomialNB

model_char_nb = MultinomialNB()

model_char_nb.fit(X_train_char, y_train)

y_pred_char_nb = model_char_nb.predict(X_test_char)

accuracy_char_nb = accuracy_score(y_test, y_pred_char_nb)

print("Character TF-IDF + Naive Bayes Accuracy:", accuracy_char_nb)

macro_f1_char_nb = f1_score(
    y_test,
    y_pred_char_nb,
    average="macro"
)

print("Character TF-IDF + Naive Bayes Macro F1:", macro_f1_char_nb)


decision_scores = model_char.decision_function(X_test_char)

print("Decision score shape:", decision_scores.shape)
print("First 5 decision scores:")
print(decision_scores[:5])



import numpy as np

sorted_scores = np.sort(decision_scores, axis=1)

margin = sorted_scores[:, -1] - sorted_scores[:, -2]

print("First 10 margins:")
print(margin[:10])


threshold = 0.2

uncertain = margin < threshold

print("Total test complaints:", len(margin))
print("Sent for human verification:", uncertain.sum())
print("Verification percentage:", uncertain.mean() * 100)


errors = y_test.values != y_pred_char

errors_caught = errors & uncertain

print("Total model errors:", errors.sum())
print("Errors caught by human verification:", errors_caught.sum())
print("Percentage of errors caught:", errors_caught.sum() / errors.sum() * 100)


threshold = 0.1

uncertain = margin < threshold

errors_caught = errors & uncertain

print("Threshold:", threshold)
print("Sent for human verification:", uncertain.sum())
print("Verification percentage:", uncertain.mean() * 100)
print("Errors caught:", errors_caught.sum())
print("Percentage of errors caught:", errors_caught.sum() / errors.sum() * 100)


threshold = 0.3

uncertain = margin < threshold

errors_caught = errors & uncertain

print("Threshold:", threshold)
print("Sent for human verification:", uncertain.sum())
print("Verification percentage:", uncertain.mean() * 100)
print("Errors caught:", errors_caught.sum())
print("Percentage of errors caught:", errors_caught.sum() / errors.sum() * 100)

threshold = 0.4

uncertain = margin < threshold

errors_caught = errors & uncertain

print("Threshold:", threshold)
print("Sent for human verification:", uncertain.sum())
print("Verification percentage:", uncertain.mean() * 100)
print("Errors caught:", errors_caught.sum())
print("Percentage of errors caught:", errors_caught.sum() / errors.sum() * 100)


threshold = 0.5

uncertain = margin < threshold

errors_caught = errors & uncertain

print("Threshold:", threshold)
print("Sent for human verification:", uncertain.sum())
print("Verification percentage:", uncertain.mean() * 100)
print("Errors caught:", errors_caught.sum())
print("Percentage of errors caught:", errors_caught.sum() / errors.sum() * 100)



corrected_predictions = y_pred_char.copy()

corrected_predictions[errors_caught] = y_test.values[errors_caught]

corrected_accuracy = accuracy_score(
    y_test,
    corrected_predictions
)

print("Simulated corrected accuracy:", corrected_accuracy)
print("Simulated corrected accuracy (%):", corrected_accuracy * 100)

simulated_macro_f1 = f1_score(
    y_test,
    corrected_predictions,
    average="macro"
)

print("Simulated corrected Macro F1:", simulated_macro_f1)





new_complaints = [
    "My hostel room fan is not working.",
    "I cannot access my university email account.",
    "I have a problem with my semester fee payment.",
    "The college bus is arriving very late."
]

new_complaints_tfidf = tfidf_char.transform(new_complaints)

new_predictions = model_char.predict(new_complaints_tfidf)

for complaint, prediction in zip(new_complaints, new_predictions):
    print("Complaint:", complaint)
    print("Predicted Department:", prediction)
    print()



from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_char))



from sklearn.pipeline import Pipeline
import joblib

final_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5)
    )),
    ("svm", LinearSVC(C=1.0))
])

final_model.fit(X_train, y_train)

joblib.dump(final_model, "issue_route_model.pkl")

print("Final model saved successfully!")
