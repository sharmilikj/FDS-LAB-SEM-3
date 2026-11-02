# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Data Collection
# Load the dataset (update path if needed)
db = pd.read_csv("/content/drive/MyDrive/PIMA Indian Dataset.csv")

# Display the first few rows
print("First 5 rows of dataset:")
print(db.head())

# Step 2: Initial Exploration
# Dataset structure and summary statistics
print("\nDataset Info:")
print(db.info())

print("\nSummary Statistics:")
print(db.describe())

# Step 3: Visualization
# Visualize the distribution of each feature
db.hist(bins=50, figsize=(20, 15))
plt.show()

# Pairplot to visualize relationships
sns.pairplot(db)
plt.show()
