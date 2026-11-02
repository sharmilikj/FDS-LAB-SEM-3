import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dataset.csv")

# 1. First 5 rows
print("FIRST 5 ROWS")
print(df.head())

# 2. Shape
print("\nDATASET SHAPE")
print(df.shape)

# 3. Information
print("\nDATASET INFORMATION")
df.info()

# 4. Missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# 5. Duplicate values
print("\nDUPLICATE VALUES")
print("Duplicates:", df.duplicated().sum())

# 6. Descriptive statistics
print("\nDESCRIPTIVE STATISTICS")
print(df.describe())

# 7. State distribution
print("\nSTATE DISTRIBUTION")
print(df['state'].value_counts())

sns.countplot(x='state', data=df)
plt.title("State Distribution")
plt.show()

# 8. Gender distribution
print("\nGENDER DISTRIBUTION")
print(df['gender'].value_counts())

sns.countplot(x='gender', data=df)
plt.title("Gender Distribution")
plt.show()

# 9. WTP distribution
print("\nWTP DISTRIBUTION")
print(df['wtp'].value_counts())

sns.countplot(x='wtp', data=df)
plt.title("Willingness to Pay")
plt.show()

# 10. Knowledge vs WTP
print("\nKNOWLEDGE VS WTP")
print(pd.crosstab(df['know'], df['wtp']))

sns.countplot(x='know', hue='wtp', data=df)
plt.title("Knowledge vs WTP")
plt.show()

# 11. Age vs WTP
sns.boxplot(x='wtp', y='age', data=df)
plt.title("Age vs WTP")
plt.show()

# 12. Income vs WTP
sns.boxplot(x='wtp', y='income', data=df)
plt.title("Income vs WTP")
plt.show()

# 13. Education vs WTP
sns.countplot(x='edu', hue='wtp', data=df)
plt.xticks(rotation=45)
plt.title("Education vs WTP")
plt.show()

# 14. Correlation
print("\nCORRELATION MATRIX")

num_cols = ['age', 'hhsize', 'kids', 'income', 'bill', 'eduyear']

corr = df[num_cols].corr()
print(corr)

sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.show()

# 15. Conclusion
print("\nCONCLUSION")
print("EDA was performed successfully.")
