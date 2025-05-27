import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess the dataset
df = pd.read_csv("Titanic.csv")

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Cabin'] = df['Cabin'].fillna('Unknown')

# Encode categorical variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Drop irrelevant columns
df = df.drop(columns=['Name', 'Ticket', 'Cabin'])

# Histograms for numeric features
df[['Age', 'Fare', 'SibSp', 'Parch']].hist(bins=20, figsize=(10, 6))
plt.suptitle("Histograms of Numeric Features")
plt.tight_layout()
plt.show()

# Boxplots to identify outliers
numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']
plt.figure(figsize=(10, 8))
for i, col in enumerate(numeric_cols):
    plt.subplot(2, 2, i + 1)
    plt.boxplot(df[col])
    plt.title(f'Boxplot of {col}')
    plt.ylabel(col)
plt.tight_layout()
plt.show()

# Correlation matrix
correlation = df.corr()
print("Correlation Matrix:\n", correlation)

# Correlation heatmap (matplotlib only)
plt.figure(figsize=(8, 6))
plt.imshow(correlation, cmap='coolwarm', interpolation='none', aspect='auto')
plt.colorbar()
plt.xticks(range(len(correlation)), correlation.columns, rotation=90)
plt.yticks(range(len(correlation)), correlation.columns)
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()
