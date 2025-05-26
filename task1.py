import pandas as pd
data=pd.read_csv('Titanic.csv')
print(data)
# View top rows
print(data.head())

# Check data types and non-null counts
print(data.info())

# Check for null values
print(data.isnull().sum())

# Basic statistics (optional)
print(data.describe())
