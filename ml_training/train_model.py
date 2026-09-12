import pandas as pd

df = pd.read_csv("dataset/complaints.csv")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nComplaints per department:")
print(df["department"].value_counts())

print("\nComplaints per urgency level:")
print(df["urgency"].value_counts())

print("Empty complaints:", df["text"].isna().sum())

print("\nData types:")
print(df.dtypes)

print("Departments:")
print(df["department"].unique())

print("\nUrgency levels:")
print(df["urgency"].unique())

# Save a copy for further processing
df.to_csv("../dataset/cleaned_complaints.csv", index=False)

print("Cleaned dataset saved successfully!")