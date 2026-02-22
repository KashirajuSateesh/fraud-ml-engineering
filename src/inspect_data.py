import pandas as pd

df = pd.read_csv("data/creditcard.csv")

print("Shape (rows, cols): ", df.shape)
print("\ncoloums:\n", df.columns.tolist())

print("\nFirst 5 Rows:\n")
print(df.head())

print("\nTarget Distribution (Class):")
print(df["Class"].value_counts())
print("\nFraud %:")
print(df["Class"].mean()*100)