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

data_path = "dataset/baseline_dataset.csv"

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
joblib.dump(final_model, "../model/issue_route_model.pkl")

print("Final model saved successfully!")






# =========================
# Research Dataset
# =========================

research_data_path = "../dataset/research_dataset.csv"

research_df = pd.read_csv(research_data_path)

print("\nResearch Dataset:")
print(research_df.head())

print("Research Dataset shape:", research_df.shape)
print("Research Dataset columns:", research_df.columns.tolist())

print("Shape:", research_df.shape)
print("\nComplaint types:")
print(research_df["complaint_type"].value_counts())

print("\nLanguages:")
print(research_df["language"].value_counts())

research_df = pd.read_csv("research_dataset.csv")

print("Shape:", research_df.shape)

print("\nComplaint Types:")
print(research_df["complaint_type"].value_counts())

print("\nLanguages:")
print(research_df["language"].value_counts())

print("\nPrimary Departments:")
print(research_df["primary_department"].value_counts())

print("\nSecondary Departments:")
print(research_df["secondary_department"].value_counts())

print("\nMissing Values:")
print(research_df.isnull().sum())

print("\nDuplicate Rows:", research_df.duplicated().sum())

research_df = research_df.drop_duplicates()

print("Shape after removing duplicate:", research_df.shape)
print("Duplicate Rows:", research_df.duplicated().sum())




research_df = research_df.drop_duplicates()

research_df.to_csv(
    "../dataset/research_dataset.csv",
    index=False
)

print("Dataset saved successfully!")
print("Shape:", research_df.shape)


print("Shape:", research_df.shape)

print("\nComplaint Types:")
print(research_df["complaint_type"].value_counts())

print("\nLanguages:")
print(research_df["language"].value_counts())

print("\nPrimary Departments:")
print(research_df["primary_department"].value_counts())

print("\nDuplicate Rows:", research_df.duplicated().sum())



X = research_df["text"]
y = research_df["complaint_type"]

print("Input (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)


from sklearn.svm import LinearSVC

model = LinearSVC()

model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("Predictions:")
print(y_pred[:10])

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)

from sklearn.metrics import f1_score

macro_f1 = f1_score(y_test, y_pred, average="macro")

print("Macro F1:", macro_f1)


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)


y_multi = research_df["complaint_type"].apply(
    lambda x: "multi" if x == "multi" else "not_multi"
)

print(y_multi.value_counts())





X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_multi,
    test_size=0.20,
    random_state=42,
    stratify=y_multi
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)




tfidf_multi = TfidfVectorizer()

X_train_multi_tfidf = tfidf_multi.fit_transform(X_train)
X_test_multi_tfidf = tfidf_multi.transform(X_test)

print("Training TF-IDF shape:", X_train_multi_tfidf.shape)
print("Testing TF-IDF shape:", X_test_multi_tfidf.shape)


model_multi = LinearSVC()

model_multi.fit(X_train_multi_tfidf, y_train)

y_pred_multi = model_multi.predict(X_test_multi_tfidf)

print("Predictions:")
print(y_pred_multi[:10])

accuracy_multi = accuracy_score(y_test, y_pred_multi)

print("Accuracy:", accuracy_multi)
print("Accuracy (%):", accuracy_multi * 100)



macro_f1_multi = f1_score(
    y_test,
    y_pred_multi,
    average="macro"
)

print("Macro F1:", macro_f1_multi)

print(classification_report(y_test, y_pred_multi))

cm_multi = confusion_matrix(y_test, y_pred_multi)

print(cm_multi)


X = research_df["text"]
y_department = research_df["primary_department"]

print(y_department.value_counts())






from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_department,
    test_size=0.20,
    random_state=42,
    stratify=y_department
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))



from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_department = TfidfVectorizer()

X_train_department_tfidf = tfidf_department.fit_transform(X_train)
X_test_department_tfidf = tfidf_department.transform(X_test)

print("Training TF-IDF shape:", X_train_department_tfidf.shape)
print("Testing TF-IDF shape:", X_test_department_tfidf.shape)


from sklearn.svm import LinearSVC

model_department = LinearSVC()

model_department.fit(
    X_train_department_tfidf,
    y_train
)

y_pred_department = model_department.predict(
    X_test_department_tfidf
)

print(y_pred_department[:10])


from sklearn.metrics import accuracy_score

accuracy_department = accuracy_score(
    y_test,
    y_pred_department
)

print("Accuracy:", accuracy_department)


from sklearn.metrics import f1_score

macro_f1_department = f1_score(
    y_test,
    y_pred_department,
    average="macro"
)

print("Macro F1:", macro_f1_department)


from sklearn.metrics import classification_report

print(classification_report(
    y_test,
    y_pred_department
))



cm_department = confusion_matrix(
    y_test,
    y_pred_department
)

print(cm_department)


errors = pd.DataFrame({
    "Complaint": X_test,
    "Actual": y_test,
    "Predicted": y_pred_department
})

errors = errors[errors["Actual"] != errors["Predicted"]]

print(errors.to_string(index=False))


error_pairs = errors.groupby(
    ["Actual", "Predicted"]
).size().sort_values(ascending=False)

print(error_pairs)


y_unknown = research_df["primary_department"].apply(
    lambda x: "unknown" if x == "Unknown" else "known"
)

print(y_unknown.value_counts())



X = research_df["text"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_unknown,
    test_size=0.20,
    random_state=42,
    stratify=y_unknown
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))



from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_unknown = TfidfVectorizer()

X_train_unknown_tfidf = tfidf_unknown.fit_transform(X_train)
X_test_unknown_tfidf = tfidf_unknown.transform(X_test)

print("Training TF-IDF shape:", X_train_unknown_tfidf.shape)
print("Testing TF-IDF shape:", X_test_unknown_tfidf.shape)



from sklearn.svm import LinearSVC

model_unknown = LinearSVC()

model_unknown.fit(
    X_train_unknown_tfidf,
    y_train
)

y_pred_unknown = model_unknown.predict(
    X_test_unknown_tfidf
)

print(y_pred_unknown[:10])


from sklearn.metrics import accuracy_score

accuracy_unknown = accuracy_score(
    y_test,
    y_pred_unknown
)

print("Accuracy:", accuracy_unknown)


from sklearn.metrics import f1_score

macro_f1_unknown = f1_score(
    y_test,
    y_pred_unknown,
    average="macro"
)

print("Macro F1:", macro_f1_unknown)



from sklearn.metrics import classification_report

print(classification_report(
    y_test,
    y_pred_unknown
))



from sklearn.metrics import confusion_matrix

cm_unknown = confusion_matrix(
    y_test,
    y_pred_unknown
)

print(cm_unknown)



error_mask_unknown = y_test != y_pred_unknown

errors_unknown = pd.DataFrame({
    "Complaint": X_test[error_mask_unknown],
    "Actual": y_test[error_mask_unknown],
    "Predicted": y_pred_unknown[error_mask_unknown]
})

print(errors_unknown.to_string(index=False))


print(research_df["language"].value_counts())


results = pd.DataFrame({
    "text": X_test,
    "actual": y_test,
    "predicted": y_pred_department,
    "language": research_df.loc[X_test.index, "language"]
})

print(results["language"].value_counts())

for lang in ["English", "Hindi", "Hinglish"]:
    lang_data = results[results["language"] == lang]

    accuracy = (lang_data["actual"] == lang_data["predicted"]).mean()

    print(lang, "Accuracy:", accuracy)



X = research_df["text"]
y_department = research_df["primary_department"]

X_train_dept, X_test_dept, y_train_dept, y_test_dept = train_test_split(
    X,
    y_department,
    test_size=0.20,
    random_state=42,
    stratify=y_department
)

print("Training samples:", len(X_train_dept))
print("Testing samples:", len(X_test_dept))



tfidf_dept = TfidfVectorizer()

X_train_dept_tfidf = tfidf_dept.fit_transform(X_train_dept)
X_test_dept_tfidf = tfidf_dept.transform(X_test_dept)

model_dept = LinearSVC()

model_dept.fit(
    X_train_dept_tfidf,
    y_train_dept
)

y_pred_dept = model_dept.predict(X_test_dept_tfidf)

print(y_pred_dept[:10])




results = pd.DataFrame({
    "actual": y_test_dept,
    "predicted": y_pred_dept,
    "language": research_df.loc[X_test_dept.index, "language"]
})

for lang in ["English", "Hindi", "Hinglish"]:
    lang_data = results[results["language"] == lang]

    accuracy = (lang_data["actual"] == lang_data["predicted"]).mean()

    print(lang, "Accuracy:", accuracy)



results = pd.DataFrame({
    "actual": y_test_dept,
    "predicted": y_pred_dept,
    "language": research_df.loc[X_test_dept.index, "language"]
})

for lang in ["English", "Hindi", "Hinglish"]:
    lang_data = results[results["language"] == lang]

    accuracy = (lang_data["actual"] == lang_data["predicted"]).mean()

    print(lang, "Accuracy:", accuracy)


    errors = results[results["actual"] != results["predicted"]]

print(errors.to_string(index=False))



multi_df = research_df[
    research_df["complaint_type"] == "multi"
]

print("Multi-issue complaints:", len(multi_df))
print(multi_df[[
    "text",
    "primary_department",
    "secondary_department"
]].head(10))

X_multi = multi_df["text"]
y_secondary = multi_df["secondary_department"]

print(y_secondary.value_counts())


X_train_sec, X_test_sec, y_train_sec, y_test_sec = train_test_split(
    X_multi,
    y_secondary,
    test_size=0.20,
    random_state=42,
    stratify=y_secondary
)

print("Training samples:", len(X_train_sec))
print("Testing samples:", len(X_test_sec))



tfidf_secondary = TfidfVectorizer()

X_train_sec_tfidf = tfidf_secondary.fit_transform(X_train_sec)
X_test_sec_tfidf = tfidf_secondary.transform(X_test_sec)

print("Training TF-IDF shape:", X_train_sec_tfidf.shape)
print("Testing TF-IDF shape:", X_test_sec_tfidf.shape)


model_secondary = LinearSVC()

model_secondary.fit(
    X_train_sec_tfidf,
    y_train_sec
)

y_pred_secondary = model_secondary.predict(
    X_test_sec_tfidf
)

print(y_pred_secondary)



from sklearn.metrics import accuracy_score

accuracy_secondary = accuracy_score(
    y_test_sec,
    y_pred_secondary
)

print("Secondary Department Accuracy:", accuracy_secondary)

from sklearn.metrics import f1_score

macro_f1_secondary = f1_score(
    y_test_sec,
    y_pred_secondary,
    average="macro"
)

print("Secondary Department Macro F1:", macro_f1_secondary)


from sklearn.metrics import classification_report

print(classification_report(
    y_test_sec,
    y_pred_secondary
))



from sklearn.metrics import confusion_matrix

cm_secondary = confusion_matrix(
    y_test_sec,
    y_pred_secondary
)

print(cm_secondary)
print("Class order:")
print(model_secondary.classes_)



results_secondary = pd.DataFrame({
    "Complaint": X_test_sec.values,
    "Actual Secondary": y_test_sec.values,
    "Predicted Secondary": y_pred_secondary
})

errors_secondary = results_secondary[
    results_secondary["Actual Secondary"] != results_secondary["Predicted Secondary"]
]

print(errors_secondary.to_string(index=False))


decision_scores = model_dept.decision_function(X_test_dept_tfidf)

print("Decision score shape:", decision_scores.shape)


import numpy as np

sorted_scores = np.sort(decision_scores, axis=1)

margin = sorted_scores[:, -1] - sorted_scores[:, -2]

print("Margin shape:", margin.shape)
print("First 10 margins:", margin[:10])


print("Minimum margin:", margin.min())
print("Maximum margin:", margin.max())
print("Average margin:", margin.mean())


threshold = 0.3

review_mask = margin < threshold

print("Threshold:", threshold)
print("Complaints sent for human verification:", review_mask.sum())
print("Percentage sent for review:", review_mask.mean() * 100)



errors = y_pred_dept != y_test_dept

total_errors = errors.sum()
errors_caught = (errors & review_mask).sum()

print("Total model errors:", total_errors)
print("Errors caught by human verification:", errors_caught)
print("Error detection rate:", (errors_caught / total_errors) * 100)




corrected_predictions = y_pred_dept.copy()

corrected_predictions[review_mask] = y_test_dept.values[review_mask]

corrected_accuracy = accuracy_score(
    y_test_dept,
    corrected_predictions
)

print("Accuracy after simulated human verification:", corrected_accuracy)




print("Total test complaints:", len(y_test_dept))
print("Complaints sent for review:", review_mask.sum())
print("Automatic routing:", (~review_mask).sum())
print("Review percentage:", (review_mask.mean() * 100))

macro_f1_corrected = f1_score(
    y_test_dept,
    corrected_predictions,
    average="macro"
)

print("Macro F1 after simulated human verification:", macro_f1_corrected)






from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_char = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5)
)

X_train_char = tfidf_char.fit_transform(X_train_dept)
X_test_char = tfidf_char.transform(X_test_dept)

print("Training Character TF-IDF shape:", X_train_char.shape)
print("Testing Character TF-IDF shape:", X_test_char.shape)





from sklearn.svm import LinearSVC

model_char = LinearSVC()

model_char.fit(
    X_train_char,
    y_train_dept
)

y_pred_char = model_char.predict(
    X_test_char
)

print(y_pred_char[:10])



from sklearn.metrics import accuracy_score

accuracy_char = accuracy_score(
    y_test_dept,
    y_pred_char
)

print("Character TF-IDF Accuracy:", accuracy_char)



from sklearn.metrics import f1_score

macro_f1_char = f1_score(
    y_test_dept,
    y_pred_char,
    average="macro"
)

print("Character TF-IDF Macro F1:", macro_f1_char)



from sklearn.metrics import classification_report

print(
    classification_report(
        y_test_dept,
        y_pred_char
    )
)


from sklearn.metrics import confusion_matrix

cm_char = confusion_matrix(
    y_test_dept,
    y_pred_char
)

print(cm_char)
print("Class order:")
print(model_char.classes_)



errors_char = y_pred_char != y_test_dept

print("Total Character TF-IDF errors:", errors_char.sum())

print("\nCharacter TF-IDF misclassified complaints:")
print(
    pd.DataFrame({
        "Complaint": X_test_dept[errors_char].values,
        "Actual": y_test_dept[errors_char].values,
        "Predicted": y_pred_char[errors_char]
    }).to_string(index=False)
)




thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

for threshold in thresholds:
    review_mask_temp = margin < threshold
    
    errors_caught_temp = (
        errors & review_mask_temp
    ).sum()
    
    review_count_temp = review_mask_temp.sum()
    
    error_detection_temp = (
        errors_caught_temp / total_errors
    ) * 100
    
    print(
        f"Threshold: {threshold} | "
        f"Review: {review_count_temp} | "
        f"Error Detection: {error_detection_temp:.2f}%"
    )




    for threshold in thresholds:
    review_mask_temp = margin < threshold

    corrected_temp = y_pred_dept.copy()
    corrected_temp[review_mask_temp] = y_test_dept.values[review_mask_temp]

    accuracy_temp = accuracy_score(
        y_test_dept,
        corrected_temp
    )

    print(
        f"Threshold: {threshold} | "
        f"Simulated Accuracy: {accuracy_temp:.4f}"
    )







    for threshold in thresholds:
    review_mask_temp = margin < threshold

    corrected_temp = y_pred_dept.copy()
    corrected_temp[review_mask_temp] = y_test_dept.values[review_mask_temp]

    macro_f1_temp = f1_score(
        y_test_dept,
        corrected_temp,
        average="macro"
    )

    print(
        f"Threshold: {threshold} | "
        f"Simulated Macro F1: {macro_f1_temp:.4f}"
    )



    for threshold in thresholds:
    review_mask_temp = margin < threshold

    corrected_temp = y_pred_dept.copy()
    corrected_temp[review_mask_temp] = y_test_dept.values[review_mask_temp]

    accuracy_temp = accuracy_score(
        y_test_dept,
        corrected_temp
    )

    macro_f1_temp = f1_score(
        y_test_dept,
        corrected_temp,
        average="macro"
    )

    review_percentage_temp = review_mask_temp.mean() * 100

    print(
        f"Threshold: {threshold} | "
        f"Review: {review_mask_temp.sum()} "
        f"({review_percentage_temp:.2f}%) | "
        f"Accuracy: {accuracy_temp:.4f} | "
        f"Macro F1: {macro_f1_temp:.4f}"
    )




    feature_names = tfidf_dept.get_feature_names_out()
coefficients = model_dept.coef_

print("Number of features:", len(feature_names))
print("Number of classes:", len(model_dept.classes_))
print("Classes:", model_dept.classes_)



for i, department in enumerate(model_dept.classes_):
    top_indices = coefficients[i].argsort()[-10:][::-1]

    print(f"\n{department}:")
    print(feature_names[top_indices])





sample_text = X_test_dept.iloc[0]

sample_vector = tfidf_dept.transform([sample_text])

predicted_department = model_dept.predict(sample_vector)[0]

feature_values = sample_vector.toarray()[0]
class_index = list(model_dept.classes_).index(predicted_department)

contributions = feature_values * coefficients[class_index]

top_indices = contributions.argsort()[-5:][::-1]

print("Complaint:", sample_text)
print("Predicted Department:", predicted_department)
print("Top contributing words:")

for index in top_indices:
    if feature_values[index] > 0:
        print(
            feature_names[index],
            "->",
            round(contributions[index], 4)
        )




sample_text = "My university portal is not allowing me to login."

sample_vector = tfidf_dept.transform([sample_text])

predicted_department = model_dept.predict(sample_vector)[0]

feature_values = sample_vector.toarray()[0]
class_index = list(model_dept.classes_).index(predicted_department)

contributions = feature_values * coefficients[class_index]

top_indices = contributions.argsort()[-5:][::-1]

print("Complaint:", sample_text)
print("Predicted Department:", predicted_department)
print("Top contributing words:")

for index in top_indices:
    if feature_values[index] > 0:
        print(
            feature_names[index],
            "->",
            round(contributions[index], 4)
        )



sample_text = "There is a water leakage in my hostel bathroom."

sample_vector = tfidf_dept.transform([sample_text])

predicted_department = model_dept.predict(sample_vector)[0]

feature_values = sample_vector.toarray()[0]
class_index = list(model_dept.classes_).index(predicted_department)

contributions = feature_values * coefficients[class_index]

top_indices = contributions.argsort()[-5:][::-1]

print("Complaint:", sample_text)
print("Predicted Department:", predicted_department)
print("Top contributing words:")

for index in top_indices:
    if feature_values[index] > 0:
        print(
            feature_names[index],
            "->",
            round(contributions[index], 4)
        )


sample_text = "There is water leakage in the bathroom."

sample_vector = tfidf_dept.transform([sample_text])

predicted_department = model_dept.predict(sample_vector)[0]

feature_values = sample_vector.toarray()[0]
class_index = list(model_dept.classes_).index(predicted_department)

contributions = feature_values * coefficients[class_index]

top_indices = contributions.argsort()[-5:][::-1]

print("Complaint:", sample_text)
print("Predicted Department:", predicted_department)
print("Top contributing words:")

for index in top_indices:
    if feature_values[index] > 0:
        print(
            feature_names[index],
            "->",
            round(contributions[index], 4)
        )




print(research_df[["text", "language"]].sample(10, random_state=42).to_string(index=False))




tfidf_char_multi = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5)
)

X_train_char_multi = tfidf_char_multi.fit_transform(X_train_dept)
X_test_char_multi = tfidf_char_multi.transform(X_test_dept)

model_char_multi = LinearSVC()
model_char_multi.fit(X_train_char_multi, y_train_dept)

y_pred_char_multi = model_char_multi.predict(X_test_char_multi)

print("Overall Accuracy:",
      accuracy_score(y_test_dept, y_pred_char_multi))

print("Overall Macro F1:",
      f1_score(y_test_dept, y_pred_char_multi, average="macro"))






test_languages = research_df.loc[
    X_test_dept.index, "language"
]

for language in ["English", "Hindi", "Hinglish"]:
    mask = test_languages == language

    language_accuracy = accuracy_score(
        y_test_dept[mask],
        y_pred_char_multi[mask]
    )

    print(
        f"{language} Accuracy: "
        f"{language_accuracy:.2%} "
        f"({mask.sum()} samples)"
    )



    multi_df = research_df[
    research_df["complaint_type"] == "multi"
].copy()

multi_df["target_departments"] = (
    multi_df["primary_department"]
    + " + "
    + multi_df["secondary_department"]
)

print("Multi-issue complaints:", len(multi_df))
print("\nTarget department combinations:")
print(multi_df["target_departments"].value_counts())



multi_df["department_pair"] = multi_df.apply(
    lambda row: " + ".join(
        sorted([
            row["primary_department"],
            row["secondary_department"]
        ])
    ),
    axis=1
)

print("Unique department pairs:", multi_df["department_pair"].nunique())

print("\nNormalized department pairs:")
print(multi_df["department_pair"].value_counts())



pair_counts = multi_df["department_pair"].value_counts()

print("Minimum samples in a pair:", pair_counts.min())
print("Maximum samples in a pair:", pair_counts.max())
print("Pairs with fewer than 5 samples:")
print(pair_counts[pair_counts < 5])




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

print(
    multi_df[departments].sum().sort_values(ascending=False)
)



X_multi = multi_df["text"]

Y_multi = multi_df[departments]

print("X shape:", X_multi.shape)
print("Y shape:", Y_multi.shape)
print("\nTarget columns:")
print(Y_multi.columns.tolist())


X_train_multi, X_test_multi, Y_train_multi, Y_test_multi = train_test_split(
    X_multi,
    Y_multi,
    test_size=0.20,
    random_state=42
)

print("Training samples:", len(X_train_multi))
print("Testing samples:", len(X_test_multi))
print("Training target shape:", Y_train_multi.shape)
print("Testing target shape:", Y_test_multi.shape)



tfidf_multi = TfidfVectorizer()

X_train_multi_tfidf = tfidf_multi.fit_transform(X_train_multi)
X_test_multi_tfidf = tfidf_multi.transform(X_test_multi)

print("Training TF-IDF shape:", X_train_multi_tfidf.shape)
print("Testing TF-IDF shape:", X_test_multi_tfidf.shape)



from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

multi_model = OneVsRestClassifier(LinearSVC())

multi_model.fit(
    X_train_multi_tfidf,
    Y_train_multi
)

Y_pred_multi = multi_model.predict(
    X_test_multi_tfidf
)

print("Prediction shape:", Y_pred_multi.shape)



from sklearn.metrics import accuracy_score

multi_accuracy = accuracy_score(
    Y_test_multi,
    Y_pred_multi
)

print("Multi-label accuracy:", multi_accuracy)


from sklearn.metrics import hamming_loss

multi_hamming = hamming_loss(
    Y_test_multi,
    Y_pred_multi
)

print("Hamming Loss:", multi_hamming)


from sklearn.metrics import f1_score

multi_f1 = f1_score(
    Y_test_multi,
    Y_pred_multi,
    average="macro"
)

print("Multi-label Macro F1:", multi_f1)




from sklearn.metrics import classification_report

print(
    classification_report(
        Y_test_multi,
        Y_pred_multi,
        target_names=departments,
        zero_division=0
    )
)





for i in range(len(X_test_multi)):
    actual = [
        departments[j]
        for j in range(len(departments))
        if Y_test_multi.iloc[i, j] == 1
    ]

    predicted = [
        departments[j]
        for j in range(len(departments))
        if Y_pred_multi[i, j] == 1
    ]

    if actual != predicted:
        print("\nComplaint:", X_test_multi.iloc[i])
        print("Actual:   ", actual)
        print("Predicted:", predicted)




import numpy as np

false_positives = np.sum(
    (Y_pred_multi == 1) & (Y_test_multi.values == 0)
)

false_negatives = np.sum(
    (Y_pred_multi == 0) & (Y_test_multi.values == 1)
)

print("False Positives:", false_positives)
print("False Negatives:", false_negatives)


decision_multi = multi_model.decision_function(
    X_test_multi_tfidf
)

print("Decision score shape:", decision_multi.shape)
print("Minimum score:", decision_multi.min())
print("Maximum score:", decision_multi.max())


for i in range(len(X_test_multi)):
    missed = []

    for j, department in enumerate(departments):
        if Y_test_multi.iloc[i, j] == 1 and Y_pred_multi[i, j] == 0:
            missed.append(
                (department, round(decision_multi[i, j], 4))
            )

    if missed:
        print("\nComplaint:", X_test_multi.iloc[i])
        print("Missed departments:", missed)



thresholds = [-0.5, -0.4, -0.3, -0.2, -0.1, 0]

for threshold in thresholds:

    Y_pred_threshold = (
        decision_multi >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        Y_test_multi,
        Y_pred_threshold
    )

    hamming = hamming_loss(
        Y_test_multi,
        Y_pred_threshold
    )

    f1 = f1_score(
        Y_test_multi,
        Y_pred_threshold,
        average="macro"
    )

    print(
        f"Threshold: {threshold:.1f} | "
        f"Subset Accuracy: {accuracy:.4f} | "
        f"Hamming Loss: {hamming:.4f} | "
        f"Macro F1: {f1:.4f}"
    )



Y_pred_best = (
    decision_multi >= -0.2
).astype(int)

false_positives_best = np.sum(
    (Y_pred_best == 1) & (Y_test_multi.values == 0)
)

false_negatives_best = np.sum(
    (Y_pred_best == 0) & (Y_test_multi.values == 1)
)

print("False Positives at -0.2:", false_positives_best)
print("False Negatives at -0.2:", false_negatives_best)