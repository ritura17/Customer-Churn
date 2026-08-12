import pandas as pd

file_path = "data/raw/Customer_Churn.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset Shape:", df.shape)