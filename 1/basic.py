


# 1. Perform the following operations using Python on a data set : read data
# from different formats(like csv, xls),indexing and selecting data, sort data,
# describe attributes of data, checking data types of each column. (Use
# Titanic Dataset).



#1

import pandas as pd

# Load Titanic dataset from a CSV file
titanic_csv = r"D:\TY 2024-25\DSML\Lab\Exam\Datasets(1)\Datasets\Titanic.csv"
data_csv = pd.read_csv(titanic_csv)

# Load Titanic dataset from an Excel file
# Note: Replace 'path_to_file.xlsx' with the actual path of the Excel file if available.
# data_xls = pd.read_excel('path_to_file.xlsx')

# Viewing the first few rows of the dataset
print("First 5 rows of the data:\n", data_csv.head())

# Viewing the last few rows of the dataset
print("First 5 rows of the data:\n", data_csv.tail())

# Viewing rows in between of the dataset
print("First 5 rows of the data:\n", data_csv.iloc[10:30])

# Viewing the first few rows of the dataset
print("First 10 rows of the data:\n", data_csv.head(10))

# Indexing and selecting data
# Select the 'Name' column
print("\nFirst 5 names in the dataset:\n", data_csv['Name'].head())

# Select rows where Age > 30
print("\nRows where Age > 30:\n", data_csv[data_csv['Age'] > 30].head())

# Sorting the data by 'Age'
sorted_data = data_csv.sort_values(by='Age', ascending=True)
print("\nData sorted by Age (first 5 rows):\n", sorted_data.head())

# Describing attributes of the data
print("\nStatistical description of the data:\n", data_csv.describe())

# Checking data types of each column
print("\nData types of each column:\n", data_csv.dtypes)


















# 24. Perform the following operations using Python on a suitable data set,
# counting unique values of data, format of each column, converting variable
# data type (e.g. from long to short, vice versa), identifying missing values
# and filling in the missing values.




#24

import pandas as pd
import numpy as np

# Example DataFrame for demonstration
data = {
    'Age': [25, 30, 22, np.nan, 29, np.nan, 35],
    'Income': ['High', 'Medium', 'Low', 'Medium', 'High', 'Low', 'Medium'],
    'Married': ['No', 'Yes', 'No', 'Yes', 'Yes', 'No', np.nan],
    'Health': ['Good', 'Fair', 'Fair', 'Good', 'Fair', 'Good', 'Fair'],
    'Salary': [50000, 60000, 40000, 55000, np.nan, 65000, 70000]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print("Original DataFrame:")
print(df)
print("\n")

# 1. Counting unique values of data
print("Unique values in each column:")
print(df.nunique())  # Count of unique values per column
print("\n")

# 2. Format of each column (data types)
print("Data types of each column:")
print(df.dtypes)  # Data types of each column
print("\n")

# 3. Converting variable data types
# Example: Convert 'Salary' from float to integer
df['Salary'] = df['Salary'].fillna(0).astype(int)  # First filling NaN with 0 before conversion
print("Data types after converting 'Salary' to int:")
print(df.dtypes)
print("\n")

# Example: Convert 'Age' from float to integer (after filling missing values)
df['Age'] = df['Age'].fillna(df['Age'].mean()).astype(int)  # Filling NaN with mean
print("Data types after converting 'Age' to int:")
print(df.dtypes)
print("\n")

# 4. Identifying missing values
print("Missing values in each column:")
print(df.isnull().sum())  # Count of missing values in each column
print("\n")

# 5. Filling in the missing values
# Filling missing values using mean for numeric columns and mode for categorical columns
df['Age'] = df['Age'].fillna(df['Age'].mean())  # Fill NaN in 'Age' with mean
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())  # Fill NaN in 'Salary' with mean
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])  # Fill NaN in 'Married' with mode
df['Income'] = df['Income'].fillna(df['Income'].mode()[0])  # Fill NaN in 'Income' with mode
df['Health'] = df['Health'].fillna(df['Health'].mode()[0])  # Fill NaN in 'Health' with mode

print("DataFrame after filling missing values:")
print(df)
print("\n")















# 25. Perform Data Cleaning, Data transformation using Python on any data
# set



#25

import pandas as pd
import numpy as np

# Sample data
data = {
    'Age': [25, 30, 22, np.nan, 29, np.nan, 35, 30, 22],
    'Income': ['High', 'Medium', 'Low', 'Medium', 'High', 'Low', 'Medium', 'Medium', np.nan],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
    'Salary': [50000, 60000, 40000, 55000, np.nan, 65000, 70000, 70000, np.nan],
    'City': ['New York', 'Los Angeles', 'New York', 'Chicago', 'Los Angeles', 'Chicago', 'New York', 'Los Angeles', 'Chicago']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Display the original DataFrame
print("Original DataFrame:")
print(df)
print("\n")

# Data Cleaning

# 1. Handling missing values:
# For numerical columns, we'll fill missing values with the mean of that column.
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())
# For categorical columns, we'll fill missing values with the mode (most frequent value).
df['Income'] = df['Income'].fillna(df['Income'].mode()[0])

# 2. Removing duplicates
df = df.drop_duplicates()

# 3. Handling invalid entries
# Let's say we consider any salary greater than 100,000 to be an invalid entry and should be removed.
df = df[df['Salary'] <= 100000]

# 4. Ensure consistency in format
# Convert "Income" to a consistent format by capitalizing the first letter of each entry.
df['Income'] = df['Income'].str.capitalize()

# Display cleaned DataFrame
print("Cleaned DataFrame:")
print(df)
print("\n")

# Data Transformation

# 1. Convert 'Age' to an integer type (after filling missing values)
df['Age'] = df['Age'].astype(int)

# 2. Normalize the 'Salary' column using Min-Max scaling (for better analysis or modeling)
df['Salary'] = (df['Salary'] - df['Salary'].min()) / (df['Salary'].max() - df['Salary'].min())

# 3. Encode categorical columns like 'Income' and 'Gender' using label encoding.
df['Income'] = df['Income'].map({'High': 3, 'Medium': 2, 'Low': 1})  # Simple label encoding for 'Income'
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})  # Simple label encoding for 'Gender'

# Display transformed DataFrame
print("Transformed DataFrame:")
print(df)
