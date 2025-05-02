import pandas as pd
import numpy as np

# Task 1: Load and Explore the Dataset

# Simulate a CSV file (replace with your actual file path)
data = {
    'employee_id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'department': ['Sales', 'Marketing', 'Sales', 'Engineering', None],
    'salary': [50000, 60000, np.nan, 75000, 55000],
    'hire_date': ['2023-01-15', '2022-05-20', '2023-03-10', None, '2022-11-01']
}

file_path = 'employees.csv'
df = pd.DataFrame(data)
df.to_csv(file_path, index=False)

try:
    # Load the dataset
    employee_df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")

    # Display the first few rows
    print("\nFirst 5 rows of the dataset:")
    print(employee_df.head())

    # Explore the structure
    print("\nDataset information:")
    employee_df.info()

    # Check for missing values
    print("\nMissing values:")
    print(employee_df.isnull().sum())

    # Clean the dataset (handling missing values)
    print("\nCleaning the dataset:")

    # Fill missing 'department' with 'Unknown'
    employee_df['department'].fillna('Unknown', inplace=True)
    print("Missing 'department' values filled with 'Unknown'.")

    # Fill missing 'salary' with the mean salary
    mean_salary = employee_df['salary'].mean()
    employee_df['salary'].fillna(mean_salary, inplace=True)
    print(f"Missing 'salary' values filled with the mean salary: {mean_salary:.2f}")

    # For 'hire_date', let's fill missing values with a placeholder or the most frequent date
    # Here, I'll fill with a placeholder
    employee_df['hire_date'].fillna('Unknown', inplace=True)
    print("Missing 'hire_date' values filled with 'Unknown'.")

    # Verify that missing values have been handled
    print("\nMissing values after cleaning:")
    print(employee_df.isnull().sum())

    print("\nFirst 5 rows after cleaning:")
    print(employee_df.head())

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")


# Task 2: Data Analysis and Visualization
# Task 2: Basic Data Analysis

# Compute basic statistics of numerical columns
print("\nBasic statistics of numerical columns:")
print(employee_df.describe())

# Perform groupings on a categorical column ('department') and compute the mean of a numerical column ('salary') for each group
print("\nMean salary per department:")
print(employee_df.groupby('department')['salary'].mean())

print("\nInteresting findings:")
print("- The average salary varies across different departments.")
print("- The 'Engineering' department seems to have the highest average salary in this small sample.")
print("- The statistics from .describe() give us a sense of the salary distribution, such as the range and central tendency.")




# Task 3: Data Visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset (if you haven't already)
try:
    iris_df = pd.read_csv('iris.csv')  # Assuming you have 'iris.csv' in the same directory
except FileNotFoundError:
    print("Error: 'iris.csv' not found. Please make sure the file is in the correct directory.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    exit()

# Add a dummy time-based column for the line chart demonstration
# In a real-world scenario, you would have an actual date or time column
iris_df['sample_index'] = iris_df.index

# 1. Line chart (showing a trend - here, sepal length over the sample index)
plt.figure(figsize=(10, 6))
plt.plot(iris_df['sample_index'].head(50), iris_df['sepal length (cm)'].head(50), marker='o', linestyle='-')
plt.title('Sepal Length of First 50 Iris Samples')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.grid(True)
plt.show()

# 2. Bar chart (comparison of average petal length per species)
average_petal_length = iris_df.groupby('target')['petal length (cm)'].mean()
plt.figure(figsize=(8, 6))
average_petal_length.plot(kind='bar', color=['skyblue', 'lightcoral', 'lightgreen'])
plt.title('Average Petal Length per Iris Species')
plt.xlabel('Iris Species')
plt.ylabel('Average Petal Length (cm)')
plt.xticks(rotation=0)  # Keep x-axis labels horizontal
plt.show()

# 3. Histogram (distribution of sepal width)
plt.figure(figsize=(8, 6))
sns.histplot(iris_df['sepal width (cm)'], bins=10, kde=True, color='orange')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

# 4. Scatter plot (relationship between sepal length and petal length, colored by species)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='target', data=iris_df)
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Iris Species')
plt.grid(True)
plt.show()