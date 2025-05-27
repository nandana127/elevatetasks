import pandas as pd

# Load the dataset
df = pd.read_csv("Titanic.csv")

# Handle missing values (optional but recommended before statistics)
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Generate summary statistics using describe()
print("Summary statistics (mean, std, min, max, etc.):\n")
print(df.describe())

# Additionally, you can print median separately
print("\nMedian of numerical columns:\n")
print(df.median(numeric_only=True))
