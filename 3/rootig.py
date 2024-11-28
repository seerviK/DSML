







# 4. Write a program to do: A dataset collected in a cosmetics shop showing
# details of customers and whether or not they responded to a special offer
# to buy a new lip-stick is shown in table below. (Implement step by step
# using commands - Dont use library) Use this dataset to build a decision
# tree, with Buys as the target variable, to help in buying lipsticks in the
# future. Find the root node of the decision tree.







#4

features = ['Age', 'Income', 'Gender', 'Ms']

import math
import pandas as pd

# Load the dataset from the CSV file
dataset_path = "/content/Lipstick.csv"  # Update with your actual file path
data = pd.read_csv(dataset_path)

# Helper function to calculate entropy
def calculate_entropy(data, target_attribute):
    target_values = data[target_attribute].tolist()  # Convert the column to a list
    total_instances = len(target_values)
    value_counts = {value: target_values.count(value) for value in set(target_values)}

    entropy = 0
    for count in value_counts.values():
        probability = count / total_instances
        entropy -= probability * math.log2(probability)
    return entropy

# Helper function to calculate information gain
def calculate_information_gain(data, feature, target_attribute):
    total_entropy = calculate_entropy(data, target_attribute)
    feature_values = data[feature].tolist()  # Convert the column to a list
    total_instances = len(feature_values)

    # Split data by unique feature values
    value_counts = {value: feature_values.count(value) for value in set(feature_values)}

    # Weighted entropy for each subset
    weighted_entropy = 0
    for value, count in value_counts.items():
        subset = data[data[feature] == value]
        subset_entropy = calculate_entropy(subset, target_attribute)
        weighted_entropy += (count / total_instances) * subset_entropy

    # Information gain
    information_gain = total_entropy - weighted_entropy
    return information_gain

# Specify the target attribute and features
target_attribute = 'Buys'  # Replace with your actual target column name
features = ['Age', 'Income', 'Gender', 'Ms']  # Replace with your actual feature column names

# Calculate entropy of the target attribute
print(f"Entropy of the target attribute '{target_attribute}': {calculate_entropy(data, target_attribute)}\n")

# Calculate information gain for each feature
information_gains = {}
for feature in features:
    info_gain = calculate_information_gain(data, feature, target_attribute)
    information_gains[feature] = info_gain
    print(f"Information Gain for feature '{feature}': {info_gain}")

# Determine the root node
root_node = max(information_gains, key=information_gains.get)
print(f"\nThe root node is: {root_node}")








































# 23. With reference to Table , obtain the Frequency table for the
# attribute age. From the frequency table you have obtained, calculate the
# information gain of the frequency table while splitting on Age. (Use step
# by step Python/Pandas commands)


# Table


#23

import pandas as pd
import numpy as np

# Create the dataset
data = {
    'Age': ['Young', 'Young', 'Middle', 'Old', 'Old', 'Old', 'Middle', 'Young', 'Young', 'Old', 'Young', 'Middle', 'Middle', 'Old'],
    'Income': ['High', 'High', 'High', 'Medium', 'Low', 'Low', 'Low', 'Medium', 'Low', 'Medium', 'Medium', 'Medium', 'High', 'Medium'],
    'Married': ['No', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No'],
    'Health': ['Fair', 'Good', 'Fair', 'Fair', 'Fair', 'Good', 'Good', 'Fair', 'Fair', 'Fair', 'Good', 'Good', 'Fair', 'Good'],
    'Class': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Frequency table for 'Age' with respect to 'Class'
freq_table = pd.crosstab(df['Age'], df['Class'])
print("Frequency Table for 'Age':")
print(freq_table)

# Calculate the entropy of the dataset (before any split)
def entropy(class_values):
    class_counts = class_values.value_counts(normalize=True)
    return -np.sum(class_counts * np.log2(class_counts))

# Total entropy before split (considering the whole dataset)
total_entropy = entropy(df['Class'])
print(f"\nTotal Entropy: {total_entropy}")

# Calculate the weighted entropy for each age group
weighted_entropy = 0
for age in df['Age'].unique():
    subset = df[df['Age'] == age]
    subset_entropy = entropy(subset['Class'])
    weight = len(subset) / len(df)
    weighted_entropy += weight * subset_entropy

print(f"\nWeighted Entropy after splitting on 'Age': {weighted_entropy}")

# Calculate information gain
information_gain = total_entropy - weighted_entropy
print(f"\nInformation Gain from splitting on 'Age': {information_gain}")
