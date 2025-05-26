import pandas as pd

# Load the cleaned Titanic dataset
df = pd.read_csv("Titanic.csv")

# Fill missing values as done before
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Cabin'] = df['Cabin'].fillna('Unknown')

# Encode 'Sex' using label encoding: male=1, female=0
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

# Encode 'Embarked' using one-hot encoding
df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked')

# Optional: simplify 'Cabin' to just deck letter (e.g., 'C85' → 'C')
df['Cabin'] = df['Cabin'].apply(lambda x: x[0] if x != 'Unknown' else 'U')
df = pd.get_dummies(df, columns=['Cabin'], prefix='Cabin')

# Drop non-useful or text-heavy columns (optional)
df.drop(['Name', 'Ticket'], axis=1, inplace=True)

# Display the head of the encoded DataFrame
print(df.head())
