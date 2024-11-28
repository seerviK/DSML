# 11. Use Iris flower dataset and perform following :
# 1. List down the features and their types (e.g., numeric, nominal)
# available in the dataset. 2. Create a histogram for each feature in the
# dataset to illustrate the feature distributions.









#11
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset from the CSV file
data = pd.read_csv(r'D:\TY 2024-25\DSML\Lab\Exam\Datasets(1)\Datasets\IRIS.csv')

# 1. List down the features and their types
features = data.columns
feature_types = data.dtypes

print("Features and their types:")
for feature, dtype in zip(features, feature_types):
    print(f"{feature}: {dtype}")

# 2. Create a histogram for each feature in the dataset
for feature in data.columns[:-1]:  # Exclude the 'species' column for histograms
    plt.figure(figsize=(8, 4))
    sns.histplot(data[feature], kde=True)
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()






#     Histogram of Sepal Length
# The histogram of sepal length shows a distribution of the lengths of the sepals in the dataset. The distribution appears to be roughly normal with a slight skew to the right. Most of the sepal lengths are concentrated between 4.5 cm and 6.5 cm, with fewer instances of very short or very long sepals. This suggests that the majority of the Iris flowers in the dataset have sepal lengths within this range.

# Histogram of Sepal Width
# The histogram of sepal width shows a distribution of the widths of the sepals in the dataset. The distribution is somewhat normal but with a noticeable peak around 3 cm. Most of the sepal widths are concentrated between 2.5 cm and 3.5 cm. There are fewer instances of very narrow or very wide sepals. This indicates that the majority of the Iris flowers have sepal widths within this range.

# Histogram of Petal Length
# The histogram of petal length shows a distribution of the lengths of the petals in the dataset. The distribution is bimodal, with two distinct peaks. One peak is around 1.5 cm, and the other is around 4.5 cm. This suggests that there are two groups of Iris flowers with distinctly different petal lengths, which could correspond to different species of Iris.

# Histogram of Petal Width
# The histogram of petal width shows a distribution of the widths of the petals in the dataset. Similar to the petal length, the distribution is bimodal with two distinct peaks. One peak is around 0.2 cm, and the other is around 1.5 cm. This indicates that there are two groups of Iris flowers with distinctly different petal widths, likely corresponding to different species of Iris.

# Summary
# The histograms reveal that the sepal length and sepal width features have relatively normal distributions, while the petal length and petal width features have bimodal distributions. The bimodal distributions of petal length and petal width suggest that these features are particularly useful for distinguishing between different species of Iris flowers.



























# 12.Use Iris flower dataset and perform following :
# 1. Create a box plot for each feature in the dataset.
# 2. Identify and discuss distributions and identify outliers from them.

#12
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset from the CSV file
data = pd.read_csv(r'D:\TY 2024-25\DSML\Lab\Exam\Datasets(1)\Datasets\IRIS.csv')

# 1. Create a box plot for each feature in the dataset
for feature in data.columns[:-1]:  # Exclude the 'species' column for box plots
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=data[feature])
    plt.title(f'Box Plot of {feature}')
    plt.xlabel(feature)
    plt.grid(True)
    plt.show()





# To identify and discuss the distributions and outliers from the box plots of the Iris dataset, we need to analyze each feature individually. Here is the detailed analysis:

# Box Plot of Sepal Length
# Distribution: The distribution of sepal lengths is slightly skewed to the right. The median sepal length is around 5.8 cm, and the interquartile range (IQR) is between approximately 5.1 cm and 6.4 cm.
# Outliers: There are a few outliers on the lower end, indicating some flowers have unusually short sepals. These outliers could be due to natural variations in the species or measurement errors.

# Box Plot of Sepal Width
# Distribution: The distribution of sepal widths is relatively symmetric. The median sepal width is around 3.0 cm, and the IQR is between approximately 2.8 cm and 3.3 cm.
# Outliers: There are several outliers on both the lower and upper ends, indicating some flowers have unusually narrow or wide sepals. These outliers might represent natural variations or measurement errors.

# Box Plot of Petal Length
# Distribution: The distribution of petal lengths is bimodal, with two distinct peaks. One peak is around 1.5 cm, and the other is around 4.5 cm. The median petal length is around 4.35 cm, and the IQR is between approximately 1.6 cm and 5.1 cm.
# Outliers: There are several outliers on the lower end, indicating some flowers have unusually short petals. The bimodal distribution suggests the presence of different species with distinct petal lengths.

# Box Plot of Petal Width
# Distribution: The distribution of petal widths is also bimodal, with two distinct peaks. One peak is around 0.2 cm, and the other is around 1.5 cm. The median petal width is around 1.3 cm, and the IQR is between approximately 0.3 cm and 1.8 cm.
# Outliers: There are several outliers on the lower end, indicating some flowers have unusually narrow petals. The bimodal distribution suggests the presence of different species with distinct petal widths.

# Summary
# Sepal Length and Sepal Width: Both features have relatively symmetric distributions with a few outliers. The outliers could be due to natural variations or measurement errors.
# Petal Length and Petal Width: Both features have bimodal distributions with several outliers. The bimodal distributions indicate the presence of different species with distinct petal characteristics. The outliers could represent natural variations or measurement errors.
# Outliers in all features should be investigated further to understand their impact on the analysis. They might represent natural variations within the species or could be due to errors in data collection.

























# 13. Use the covid_vaccine_statewise.csv dataset and perform the following
# analytics.
# a. Describe the dataset
# b. Number of persons state wise vaccinated for first dose in India
# c. Number of persons state wise vaccinated for second dose in India


#13

import pandas as pd

# Load the dataset
dataset_path = "/content/Covid Vaccine Statewise.csv"  # Replace with your dataset path
data = pd.read_csv(dataset_path)

# a. Describe the dataset
print("Dataset Information:")
print(data.info())
print("\nDataset Summary Statistics:")
print(data.describe(include='all'))
print("\nFirst few rows of the dataset:")
print(data.head())

# b. Number of persons state-wise vaccinated for the first dose in India
# Assuming 'First Dose Administered' column exists in the dataset
statewise_first_dose = data.groupby('State')['First Dose Administered'].sum().reset_index()
print("\nNumber of persons state-wise vaccinated for the first dose:")
print(statewise_first_dose)

# c. Number of persons state-wise vaccinated for the second dose in India
# Assuming 'Second Dose Administered' column exists in the dataset
statewise_second_dose = data.groupby('State')['Second Dose Administered'].sum().reset_index()
print("\nNumber of persons state-wise vaccinated for the second dose:")
print(statewise_second_dose)

# Optional: Save the results to CSV files
statewise_first_dose.to_csv("/content/statewise_first_dose.csv", index=False)
statewise_second_dose.to_csv("/content/statewise_second_dose.csv", index=False)




























# 14. Use the covid_vaccine_statewise.csv dataset and perform the following
# analytics.
# A. Describe the dataset.
# B. Number of Males vaccinated
# C.. Number of females vaccinated




#14

import pandas as pd

# Load the dataset
dataset_path = "/content/Covid Vaccine Statewise.csv"  # Replace with your dataset path
data = pd.read_csv(dataset_path)

# A. Describe the dataset
print("Dataset Information:")
print(data.info())
print("\nDataset Summary Statistics:")
print(data.describe(include='all'))
print("\nFirst few rows of the dataset:")
print(data.head())

# B. Number of Males vaccinated
# Assuming the dataset contains a column named 'Male(Individuals Vaccinated)'
total_males_vaccinated = data['Male(Individuals Vaccinated)'].sum()
print(f"\nTotal number of males vaccinated: {total_males_vaccinated}")

# C. Number of Females vaccinated
# Assuming the dataset contains a column named 'Female(Individuals Vaccinated)'
total_females_vaccinated = data['Female(Individuals Vaccinated)'].sum()
print(f"\nTotal number of females vaccinated: {total_females_vaccinated}")
