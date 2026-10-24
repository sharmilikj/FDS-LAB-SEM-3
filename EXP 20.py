import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('Mall_Customers.csv')
df.info()
print(df.head())

# Pairplot of the whole dataset
sns.pairplot(df)
plt.show()

# Features: Annual Income and Spending Score
features = df.iloc[:, [3, 4]].values

# K-Means with 5 clusters
from sklearn.cluster import KMeans
model = KMeans(n_clusters=5, n_init=10)
model.fit(features)

# Add cluster labels
Final = df.iloc[:, [3, 4]].copy()
Final['label'] = model.predict(features)
print(Final.head())

# Cluster scatter plot
sns.set_style("whitegrid")
sns.FacetGrid(Final, hue="label", height=8) \
    .map(plt.scatter, "Annual Income (k$)", "Spending Score (1-100)") \
    .add_legend()
plt.show()

# Elbow method
features_el = df.iloc[:, [2, 3, 4]].values
wcss = []
for i in range(1, 10):
    model = KMeans(n_clusters=i, n_init=10)
    model.fit(features_el)
    wcss.append(model.inertia_)
plt.plot(range(1, 10), wcss)
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()