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