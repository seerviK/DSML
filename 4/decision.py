









# 5. Write a program to do: A dataset collected in a cosmetics shop showing
# details of customers and whether or not they responded to a special offer
# to buy a new lip-stick is shown in table below. (Use library commands)
# According to the decision tree you have made from the previous training
# data set, what is the decision for the test data: [Age < 21, Income = Low,
# Gender = Female, Marital Status = Married]?










#5

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
dataset_path = "/content/Lipstick.csv"  # Replace with your dataset file path
data = pd.read_csv(dataset_path)

# Show the first few rows of the dataset to understand its structure
print(data.head())

# Handle Age (convert age categories into numerical values)
def convert_age(age):
    if age == '<21':
        return 20
    elif age == '21-35':
        return 28  # Midpoint of 21-35
    elif age == '>35':
        return 40  # Arbitrary value for '>35'
    return age

data['Age'] = data['Age'].apply(convert_age)

# Encode categorical features ('Gender', 'Ms', and 'Income') into numerical values
label_encoder = LabelEncoder()

# Encode Gender (e.g., 'Female' -> 0, 'Male' -> 1)
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Encode Marital Status (e.g., 'Married' -> 1, 'Single' -> 0)
data['Ms'] = label_encoder.fit_transform(data['Ms'])

# Encode Income (e.g., 'Low' -> 0, 'Medium' -> 1, 'High' -> 2)
data['Income'] = label_encoder.fit_transform(data['Income'])

# Encode Buys (target column: e.g., 'Yes' -> 1, 'No' -> 0)
data['Buys'] = label_encoder.fit_transform(data['Buys'])

# Features and target column
features = ['Age', 'Income', 'Gender', 'Ms']
target = 'Buys'

# Split the dataset into features and target
X = data[features]
y = data[target]

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Define the test data point: [Age < 21, Income = Low, Gender = Female, Marital Status = Married]
test_data = [[20, 0, 0, 1]]  # Age=20, Income=Low (0), Gender=Female (0), Marital Status=Married (1)

# Make prediction
prediction = clf.predict(test_data)

# Decode the prediction back to the original value
prediction_decoded = label_encoder.inverse_transform(prediction)

# Print the prediction
print(f"The predicted response for the test data [Age < 21, Income = Low, Gender = Female, Marital Status = Married] is: {prediction_decoded[0]}")















# 6. Write a program to do: A dataset collected in a cosmetics shop showing
# details of customers and whether or not they responded to a special offer
# to buy a new lip-stick is shown in table below. (Use library commands)
# According to the decision tree you have made from the previous training
# data set, what is the decision for the test data: [Age > 35, Income =
# Medium, Gender = Female, Marital Status = Married]







#6

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
dataset_path = "/content/Lipstick.csv"  # Replace with your dataset file path
data = pd.read_csv(dataset_path)

# Show the first few rows of the dataset to understand its structure
print(data.head())

# Handle Age (convert age categories into numerical values)
def convert_age(age):
    if age == '<21':
        return 20
    elif age == '21-35':
        return 28  # Midpoint of 21-35
    elif age == '>35':
        return 40  # Arbitrary value for '>35'
    return age

data['Age'] = data['Age'].apply(convert_age)

# Encode categorical features ('Gender', 'Ms', and 'Income') into numerical values
label_encoder = LabelEncoder()

# Encode Gender (e.g., 'Female' -> 0, 'Male' -> 1)
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Encode Marital Status (e.g., 'Married' -> 1, 'Single' -> 0)
data['Ms'] = label_encoder.fit_transform(data['Ms'])

# Encode Income (e.g., 'Low' -> 0, 'Medium' -> 1, 'High' -> 2)
data['Income'] = label_encoder.fit_transform(data['Income'])

# Encode Buys (target column: e.g., 'Yes' -> 1, 'No' -> 0)
data['Buys'] = label_encoder.fit_transform(data['Buys'])

# Features and target column
features = ['Age', 'Income', 'Gender', 'Ms']
target = 'Buys'

# Split the dataset into features and target
X = data[features]
y = data[target]

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Define the test data point: [Age < 21, Income = Low, Gender = Female, Marital Status = Married]
test_data = [[40, 1, 0, 1]]  # Age=20, Income=Low (0), Gender=Female (0), Marital Status=Married (1)

# Make prediction
prediction = clf.predict(test_data)

# Decode the prediction back to the original value
prediction_decoded = label_encoder.inverse_transform(prediction)

# Print the prediction
print(f"The predicted response for the test data [Age < 21, Income = Low, Gender = Female, Marital Status = Married] is: {prediction_decoded[0]}")



































# 7. Write a program to do: A dataset collected in a cosmetics shop showing
# details of customers and whether or not they responded to a special offer
# to buy a new lip-stick is shown in table below. (Use library commands)
# According to the decision tree you have made from the previous training
# data set, what is the decision for the test data: [Age > 35, Income =
# Medium, Gender = Female, Marital Status = Married]?

# 7 is same as 6







#8

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
dataset_path = "/content/Lipstick.csv"  # Replace with your dataset file path
data = pd.read_csv(dataset_path)

# Show the first few rows of the dataset to understand its structure
print(data.head())

# Handle Age (convert age categories into numerical values)
def convert_age(age):
    if age == '<21':
        return 20
    elif age == '21-35':
        return 28  # Midpoint of 21-35
    elif age == '>35':
        return 40  # Arbitrary value for '>35'
    return age

data['Age'] = data['Age'].apply(convert_age)

# Encode categorical features ('Gender', 'Ms', and 'Income') into numerical values
label_encoder = LabelEncoder()

# Encode Gender (e.g., 'Female' -> 0, 'Male' -> 1)
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Encode Marital Status (e.g., 'Married' -> 1, 'Single' -> 0)
data['Ms'] = label_encoder.fit_transform(data['Ms'])

# Encode Income (e.g., 'Low' -> 0, 'Medium' -> 1, 'High' -> 2)
data['Income'] = label_encoder.fit_transform(data['Income'])

# Encode Buys (target column: e.g., 'Yes' -> 1, 'No' -> 0)
data['Buys'] = label_encoder.fit_transform(data['Buys'])

# Features and target column
features = ['Age', 'Income', 'Gender', 'Ms']
target = 'Buys'

# Split the dataset into features and target
X = data[features]
y = data[target]

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Define the test data point: [Age = 21-35, Income = Low, Gender = Male, Marital Status = Married]
# Age = 28 (converted from 21-35), Income = Low (0), Gender = Male (1), Marital Status = Married (1)
test_data = [[28, 0, 1, 1]]  # Age=28, Income=Low (0), Gender=Male (1), Marital Status=Married (1)

# Make prediction
prediction = clf.predict(test_data)

# Decode the prediction back to the original value
prediction_decoded = label_encoder.inverse_transform(prediction)

# Print the prediction
print(f"The predicted response for the test data [Age = 21-35, Income = Low, Gender = Male, Marital Status = Married] is: {prediction_decoded[0]}")