import numpy as np
import pandas as pd

# 1. Create a DataFrame using a list
list_data = [
    [1, 'Smith', 50000],
    [2, 'Jones', 60000]
]

df = pd.DataFrame(list_data)

print("DataFrame:")
print(df)

# 2. Give column names
df.columns = ['Empd', 'Name', 'Salary']

print("\nDataFrame with Column Names:")
print(df)

# 3. Display information about DataFrame
print("\nDataFrame Information:")
df.info()

# 4. Create employee DataFrame
employee_data = [
    [1, 'SREE VARSSINI K S', 5000],
    [2, 'SREEMATHI B', 6000],
    [3, 'SREYA G', 7000],
    [4, 'SREYASKARI MULLAPUDI', 5000],
    [5, 'SRI AKASH U G', 8000],
    [6, 'SRI HARSHAVARDHANAN R', 3000],
    [7, 'SRI HARSHAVARDHANAN R', 6000]
]

employee_df = pd.DataFrame(
    employee_data,
    columns=['emp id', 'name', 'salary']
)

print("\nEmployee DataFrame:")
print(employee_df)

# 5. First 5 rows
print("\nFirst 5 rows:")
print(employee_df.head())

# 6. Last 5 rows
print("\nLast 5 rows:")
print(employee_df.tail())

# 7. Information about employee DataFrame
print("\nEmployee DataFrame Information:")
employee_df.info()

# 8. Display salary column
print("\nSalary:")
print(employee_df.salary)

# 9. Type of salary column
print("\nType of Salary:")
print(type(employee_df.salary))

# 10. Mean
print("\nMean Salary:")
print(employee_df.salary.mean())

# 11. Median
print("\nMedian Salary:")
print(employee_df.salary.median())

# 12. Mode
print("\nMode Salary:")
print(employee_df.salary.mode())

# 13. Variance
print("\nSalary Variance:")
print(employee_df.salary.var())

# 14. Standard deviation
print("\nSalary Standard Deviation:")
print(employee_df.salary.std())

# 15. Statistical description
print("\nDescribe:")
print(employee_df.describe())

# 16. Description including all columns
print("\nDescribe Including All:")
print(employee_df.describe(include='all'))

# 17. Display column names
print("\nColumn Names:")
empCol = employee_df.columns
print(empCol)

# 18. Convert DataFrame into NumPy array
print("\nNumPy Array:")
emparray = employee_df.values
print(emparray)

# 19. Create DataFrame again from NumPy array
employee_DF = pd.DataFrame(
    emparray,
    columns=empCol
)

print("\nDataFrame Created from NumPy Array:")
print(employee_DF)