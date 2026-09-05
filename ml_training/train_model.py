import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

df=pd.read_csv("dataset/complaints.csv")
# print(df.head())
# print("\ninfo of the dataset:")
# print(df.info())
# print("\nshape of the dataset:")
# print(df.shape)
# print("\nmissing values:")
# print(df.isnull().sum())
# print("\ncomplaints per department:")
# print(df["department"].value_counts())

X = df["complaint"]

y = df["department"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# print("\nTraining samples:", len(X_train))
# print("Testing samples:", len(X_test))

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
# print("\nTF-IDF training shape:",X_train_tfidf.shape)
# print("\nTF-IDF testing shape:",X_test_tfidf.shape)
# Create the Logistic Regression model
model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

print("\nModel training completed!")

y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2%}")
# print("\nClassification report:")
# print(classification_report(y_test, y_pred,zero_division=0))

# Save the model
joblib.dump(model, "model/complaints_model.pkl")
print("\nModel saved successfully!")
joblib.dump(vectorizer, "model/complaints_vectorizer.pkl")
print("\nVectorizer saved successfully!")