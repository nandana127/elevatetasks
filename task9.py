import pandas as pd
import matplotlib.pyplot as plt

# Load and clean dataset
df = pd.read_csv("Titanic.csv")
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Cabin'] = df['Cabin'].fillna('Unknown')

# Encode categorical features
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Drop non-essential columns
df = df.drop(columns=['Name', 'Ticket', 'Cabin'])

# --- Pattern 1: Survival rate by Sex ---
survival_by_sex = df.groupby('Sex')['Survived'].mean()
plt.figure()
survival_by_sex.plot(kind='bar', color=['blue', 'pink'])
plt.title('Survival Rate by Sex (0=Male, 1=Female)')
plt.ylabel('Survival Rate')
plt.xticks([0, 1], ['Male', 'Female'], rotation=0)
plt.grid(True)

# --- Pattern 2: Survival rate by Pclass ---
survival_by_class = df.groupby('Pclass')['Survived'].mean()
plt.figure()
survival_by_class.plot(kind='bar', color='green')
plt.title('Survival Rate by Passenger Class')
plt.ylabel('Survival Rate')
plt.xticks([0, 1, 2], ['1st Class', '2nd Class', '3rd Class'], rotation=0)
plt.grid(True)

# --- Trend/Anomaly: Fare distribution ---
plt.figure()
plt.hist(df['Fare'], bins=30, color='orange', edgecolor='black')
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Number of Passengers')
plt.grid(True)

# --- Trend/Anomaly: Boxplot of Age to detect outliers ---
plt.figure()
plt.boxplot(df['Age'])
plt.title('Boxplot of Age')
plt.ylabel('Age')

plt.tight_layout()
plt.show()
