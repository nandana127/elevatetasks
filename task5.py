import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Titanic.csv")

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Cabin'] = df['Cabin'].fillna('Unknown')

# Encode categorical variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Drop unnecessary columns
df = df.drop(columns=['Name', 'Ticket', 'Cabin'])

# Plot boxplots using matplotlib
numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']
plt.figure(figsize=(12, 6))

for i, col in enumerate(numeric_cols):
    plt.subplot(2, 2, i+1)
    plt.boxplot(df[col])
    plt.title(f'Boxplot of {col}')
    plt.ylabel(col)

plt.tight_layout()
plt.show()

# Function to remove outliers using IQR
def remove_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    return dataframe[(dataframe[column] >= lower_limit) & (dataframe[column] <= upper_limit)]

# Remove outliers from each numerical column
for col in numeric_cols:
    df = remove_outliers_iqr(df, col)

# Output cleaned data shape and preview
print("\nShape after outlier removal:", df.shape)
print("\nPreview of cleaned data:\n")
print(df.head())
