
# Experiment 18: KNN Algorithm
# Description: Understand the KNN algorithm using the Iris dataset.

import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv('Iris_Dataset.csv')

# Display dataset information
df.info()

# Display species counts
print(df['variety'].value_counts())

# Display the first five rows
print(df.head())

# Separate features and labels
features = df.iloc[:, 1:-1].values
label = df['variety'].values

# Import required libraries
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    features, label, test_size=0.2, random_state=0
)

# Create the KNN model with 7 neighbors
model_KNN = KNeighborsClassifier(n_neighbors=7)

# Train the model
model_KNN.fit(X_train, y_train)

# Calculate accuracy scores
print("Training Accuracy:", model_KNN.score(X_train, y_train))
print("Testing Accuracy:", model_KNN.score(X_test, y_test))

# Predict the test data
y_pred = model_KNN.predict(X_test)

# Display the confusion matrix
from sklearn.metrics import confusion_matrix

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Display the classification report
from sklearn.metrics import classification_report

print("Classification Report:")
print(classification_report(y_test, y_pred))
