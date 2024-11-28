# 2. Perform the following operations using Python on the Telecom_Churn
# dataset. Compute and display summary statistics for each feature available
# in the dataset using separate commands for each statistic. (e.g. minimum
# value, maximum value, mean, range, standard deviation, variance and
# percentiles).


#2

import pandas as pd
import numpy as np

# Load Telecom_Churn dataset from a CSV file (replace the link with your dataset's file path)
telecom_churn_csv = "/content/Telecom Churn.csv"
data = pd.read_csv(telecom_churn_csv)

# Displaying the first few rows of the dataset
print("Dataset preview:\n", data.head())

# Select numeric columns for statistical operations
numeric_data = data.select_dtypes(include=[np.number])

# Compute and display summary statistics for each feature
for column in numeric_data.columns:
    print(f"\nStatistics for feature: {column}")

    # Minimum value
    print(f"Minimum value: {numeric_data[column].min()}")

    # Maximum value
    print(f"Maximum value: {numeric_data[column].max()}")

    # Mean
    print(f"Mean: {numeric_data[column].mean()}")

    # Range (Max - Min)
    feature_range = numeric_data[column].max() - numeric_data[column].min()
    print(f"Range: {feature_range}")

    # Standard Deviation
    print(f"Standard Deviation: {numeric_data[column].std()}")

    # Variance
    print(f"Variance: {numeric_data[column].var()}")

    # Percentiles (25th, 50th, and 75th)
    percentiles = numeric_data[column].quantile([0.25, 0.50, 0.75])
    print(f"Percentiles (25th, 50th, 75th):\n{percentiles}")



























# 3. Perform the following operations using Python on the data set
# House_Price Prediction dataset. Compute standard deviation, variance and
# percentiles using separate commands, for each feature. Create a histogram
# for each feature in the dataset to illustrate the feature distributions


#3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load House Price Prediction dataset (replace with your file path if available locally)
house_price_csv = "/content/House Data.csv"
data = pd.read_csv(house_price_csv)

# Preview the dataset
print("Dataset preview:\n", data.head())

# Select numeric columns for analysis
numeric_data = data.select_dtypes(include=[np.number])

# Compute statistics for each feature
for column in numeric_data.columns:
    print(f"\nStatistics for feature: {column}")

    # Standard Deviation
    std_dev = numeric_data[column].std()
    print(f"Standard Deviation: {std_dev}")

    # Variance
    variance = numeric_data[column].var()
    print(f"Variance: {variance}")

    # Percentiles (25th, 50th, 75th)
    percentiles = numeric_data[column].quantile([0.25, 0.50, 0.75])
    print(f"Percentiles (25th, 50th, 75th):\n{percentiles}")

    # Create histogram
    plt.figure(figsize=(6, 4))
    plt.hist(numeric_data[column].dropna(), bins=30, color='blue', alpha=0.7)
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()





















# 18. Use House_Price prediction dataset. Provide summary statistics (mean,
# median, minimum, maximum, standard deviation) of variables (categorical
# vs quantitative) such as- For example, if categorical variable is age groups
# and quantitative variable is income, then provide summary statistics of
# income grouped by the age groups

#18

import pandas as pd

# Load the House Price dataset
url = '/content/House Data.csv'
house_data = pd.read_csv(url)

# Display the first few rows of the dataset
print(house_data.head())

# Check unique entries in the 'price' column to identify issues
print(house_data['price'].unique())

# Clean and convert the 'price' column to numeric
# Remove 'TL', commas, and other non-numeric characters
house_data['price'] = house_data['price'].str.replace('TL', '', regex=False)
house_data['price'] = house_data['price'].str.replace(',', '', regex=False)
house_data['price'] = house_data['price'].str.extract('(\d+)', expand=False)  # Extract only numeric values
house_data['price'] = pd.to_numeric(house_data['price'], errors='coerce')  # Convert to float, set invalid entries to NaN

# Drop rows with NaN prices (optional, depending on the requirement)
house_data = house_data.dropna(subset=['price'])

# Categorical variable: district, Quantitative variable: price
grouped_stats = house_data.groupby('district')['price'].agg(['mean', 'median', 'min', 'max', 'std'])

# Rename the columns for clarity
grouped_stats.rename(columns={
    'mean': 'Mean Price',
    'median': 'Median Price',
    'min': 'Minimum Price',
    'max': 'Maximum Price',
    'std': 'Standard Deviation'
}, inplace=True)

# Display the grouped summary statistics
print(grouped_stats)

# Save to a CSV file for reference if needed
grouped_stats.to_csv('HousePrice_Summary_by_District.csv')


















# 19. Write a Python program to display some basic statistical details like
# percentile, mean, standard deviation etc (Use python and pandas
# commands) the species of ‘Iris-setosa’, ‘Iris-versicolor’ and ‘Iris-versicolor’
# of iris.csv dataset.





#19

import pandas as pd

# Load the Iris dataset
url = '/content/IRIS.csv'
iris_data = pd.read_csv(url)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(iris_data.head())

# List unique species
print("\nUnique species in the dataset:")
print(iris_data['species'].unique())

# Filter data for each species and compute descriptive statistics
for species in iris_data['species'].unique():
    print(f"\nStatistical details for {species}:")
    species_data = iris_data[iris_data['species'] == species]
    stats = species_data.describe(percentiles=[0.25, 0.5, 0.75])
    print(stats)



