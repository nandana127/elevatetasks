import pandas as pd

# Load the dataset
df = pd.read_csv("Titanic.csv")

# Check missing values before imputation
print("Missing values before imputation:\n", df.isnull().sum())

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Cabin'] = df['Cabin'].fillna('Unknown')

# Check missing values after imputation
print("\nMissing values after imputation:\n", df.isnull().sum())

# Show the filled values from each affected column
print("\nFilled 'Age' column (first 10 rows):\n", df['Age'].head(10))
print("\nFilled 'Fare' column (first 10 rows):\n", df['Fare'].head(10))
print("\nFilled 'Embarked' column (first 10 rows):\n", df['Embarked'].head(10))
print("\nFilled 'Cabin' column (first 10 rows):\n", df['Cabin'].head(10))
