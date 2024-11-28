# 15. Use the dataset 'titanic'. The dataset contains 891 rows and contains
# information about the passengers who boarded the unfortunate Titanic
# ship. Use the Seaborn library to see if we can find any patterns in the data.






# To analyze the Titanic dataset and find patterns using the Seaborn library, we can create various visualizations. Here are some common visualizations that can help us identify patterns in the data:

# Distribution of passengers by gender.
# Survival rate by gender.
# Survival rate by passenger class.
# Age distribution of passengers.
# Survival rate by age.
# Survival rate by embarkation point.



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# Load the dataset from the CSV file
data = pd.read_csv(r'D:\TY 2024-25\DSML\Lab\Exam\Datasets(1)\Datasets\Titanic.csv')



# Distribution of passengers by gender
plt.figure(figsize=(8, 6))
sns.countplot(x='Sex', data=data)
plt.title('Distribution of Passengers by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


# Analysis: This plot shows the count of male and female passengers on the Titanic.
# Inference: There were more male passengers than female passengers on the Titanic.


# Survival rate by gender
plt.figure(figsize=(8, 6))
sns.barplot(x='Sex', y='Survived', data=data)
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.show()

# Analysis: This plot shows the survival rate for male and female passengers.
# Inference: Female passengers had a significantly higher survival rate compared to male passengers. This suggests that women were given priority during the evacuation.


# Survival rate by passenger class
plt.figure(figsize=(8, 6))
sns.barplot(x='Pclass', y='Survived', data=data)
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.show()

# Analysis: This plot shows the survival rate for each passenger class (1st, 2nd, and 3rd).
# Inference: Passengers in the 1st class had the highest survival rate, followed by those in the 2nd class, with the 3rd class having the lowest survival rate. This indicates that higher-class passengers had better chances of survival.


# Age distribution of passengers
plt.figure(figsize=(8, 6))
sns.histplot(data['Age'].dropna(), kde=True)
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Analysis: This plot shows the distribution of passenger ages.
# Inference: The age distribution is roughly normal, with most passengers being between 20 and 40 years old. There are fewer passengers in the very young and older age groups.



# Survival rate by age
plt.figure(figsize=(8, 6))
sns.histplot(data[data['Survived'] == 1]['Age'].dropna(), kde=True, color='green', label='Survived')
sns.histplot(data[data['Survived'] == 0]['Age'].dropna(), kde=True, color='red', label='Not Survived')
plt.title('Survival Rate by Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Analysis: This plot shows the age distribution of passengers who survived and those who did not.
# Inference: Younger passengers, particularly children, had higher survival rates. The survival rate decreases with age, indicating that younger passengers were more likely to survive.


# Survival rate by embarkation point
plt.figure(figsize=(8, 6))
sns.barplot(x='Embarked', y='Survived', data=data)
plt.title('Survival Rate by Embarkation Point')
plt.xlabel('Embarkation Point')
plt.ylabel('Survival Rate')
plt.show()

# Analysis: This plot shows the survival rate for passengers based on their embarkation point (C = Cherbourg, Q = Queenstown, S = Southampton).
# Inference: Passengers who embarked from Cherbourg (C) had the highest survival rate, followed by those from Queenstown (Q) and Southampton (S). This could be related to the socio-economic status of passengers boarding from these locations.


# These visualizations help us identify several patterns in the Titanic dataset:

# There were more male passengers than female passengers.

# Female passengers had a higher survival rate compared to male passengers.

# Passengers in the 1st class had the highest survival rate, followed by the 2nd class, with the 3rd class having the lowest survival rate.

# The age distribution of passengers is roughly normal, with most passengers being between 20 and 40 years old.

# Younger passengers, particularly children, had higher survival rates.

# Passengers who embarked from Cherbourg had the highest survival rate, followed by those from Queenstown and Southampton.

# These insights can help us understand the factors that influenced survival rates on the Titanic.




# Explanation:

# Distribution of Passengers by Gender: This plot shows the count of male and female passengers.

# Survival Rate by Gender: This plot shows the survival rate for male and female passengers.

# Survival Rate by Passenger Class: This plot shows the survival rate for each passenger class (1st, 2nd, and 3rd).

# Age Distribution of Passengers: This plot shows the distribution of passenger ages.
# Survival Rate by Age: This plot shows the age distribution of passengers who survived and those who did not.

# Survival Rate by Embarkation Point: This plot shows the survival rate for passengers based on their embarkation point (C = Cherbourg, Q = Queenstown, S = Southampton).

# These visualizations will help us identify patterns in the Titanic dataset using the Seaborn library.






















# Q.16 Use the inbuilt dataset 'titanic'. The dataset contains 891 rows and
# contains information about the passengers who boarded the unfortunate
# Titanic ship. Write a code to check how the price of the ticket (column
# name: 'fare') for each passenger is distributed by plotting a histogram.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load the dataset from the CSV file
data = pd.read_csv(r'D:\TY 2024-25\DSML\Lab\Exam\Datasets(1)\Datasets\Titanic.csv')

# Plotting the histogram of ticket prices (fare)
plt.figure(figsize=(10, 6))
sns.histplot(data['Fare'], kde=True)
plt.title('Distribution of Ticket Prices (Fare)')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()



# Explanation:
# Importing Libraries: We import the necessary libraries: pandas for data manipulation, seaborn for visualization, and matplotlib.pyplot for plotting.

# Loading the Dataset: We load the Titanic dataset from the CSV file into a pandas DataFrame named data.

# Plotting the Histogram: We use Seaborn's histplot function to plot a histogram of the 'Fare' column. The kde=True parameter adds a Kernel Density Estimate (KDE) line to the histogram, which helps in visualizing the distribution more smoothly. We also set the title, x-axis label, y-axis label, and enable the grid for better readability.
# This code will display a histogram showing the distribution of ticket prices for passengers on the Titanic.




# Inference from the Output Distribution of Ticket Prices
# Analysis:

# The histogram shows the distribution of ticket prices (fare) for passengers on the Titanic.

# The majority of the ticket prices are concentrated at the lower end of the fare spectrum.

# There is a long tail towards the higher end, indicating that while most passengers paid lower fares, there were a few passengers who paid significantly higher fares.

# The Kernel Density Estimate (KDE) line helps to visualize the distribution more smoothly, showing a peak at the lower fare range.

# Inference:

# Skewed Distribution: The distribution of ticket prices is right-skewed, meaning that most passengers paid lower fares, and only a few passengers paid higher fares.

# Majority of Passengers: The majority of passengers paid fares in the lower range, which could indicate that a large number of passengers were in the lower classes (3rd class).

# High Fare Outliers: The presence of high fare outliers suggests that there were some passengers who paid premium prices for their tickets, likely corresponding to 1st class passengers.

# Economic Diversity: The wide range of ticket prices reflects the economic diversity of the passengers on the Titanic, with fares ranging from very low to very high.
# This analysis helps us understand the economic distribution of the passengers on the Titanic based on the ticket prices they paid.


























# 16. Use the inbuilt dataset 'titanic'. The dataset contains 891 rows and
# contains information about the passengers who boarded the unfortunate
# Titanic ship. Write a code to check how the price of the ticket (column
# name: 'fare') for each passenger is distributed by plotting a histogram.