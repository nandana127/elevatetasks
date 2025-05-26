import pandas as pd
import numpy as np

# Step 1: Load the dataset
df = pd.read_csv("Titanic.csv")

# Step 2: Basic info
print("Initial info:")
print(df.info())
print("\nMissing values before imputation:\n", df.isnull().sum())

# Step 3: Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())               # Use mean for Age
df['Fare'] = df['Fare'].fillna(df['Fare'].median())          # Use median for Fare
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Use mode for Embarked
df['Cabin'] = df['Cabin'].fillna('Unknown')                  # Fill Cabin with 'Unknown'

# Step 4: Check missing values again
print("\nMissing values after imputation:\n", df.isnull().sum())

# Step 5: Encode categorical features
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# You can drop columns like Name, Ticket, Cabin if not using them
df = df.drop(columns=['Name', 'Ticket', 'Cabin'])

# Step 6: Normalize numerical features manually (Min-Max Scaling)
numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']
for col in numerical_cols:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# Step 7: Show final dataset preview
print("\nFinal dataset preview:\n")
print(df.head())
