import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Titanic.csv")

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Select numeric columns
numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']

# Create histograms
plt.figure(figsize=(12, 6))
for i, col in enumerate(numeric_cols):
    plt.subplot(2, 4, i + 1)
    plt.hist(df[col], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

# Create boxplots
for i, col in enumerate(numeric_cols):
    plt.subplot(2, 4, i + 5)
    plt.boxplot(df[col])
    plt.title(f'Boxplot of {col}')
    plt.ylabel(col)

plt.tight_layout()
plt.show()
