# Experiment 17: Linear Regression for Salary Prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv('Salary_data.csv')

# Display dataset information
print(df.info())
print(df.describe())

# Select independent and dependent variables
features = df.iloc[:, :-1].values
labels = df.iloc[:, -1].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=0
)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Display model scores
print("Training Score:", model.score(X_train, y_train))
print("Testing Score:", model.score(X_test, y_test))

# Display coefficient and intercept
print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)

# Predict salary for test data
y_pred = model.predict(X_test)
print("Predicted Salaries:", y_pred)

# Plot the regression line
plt.scatter(features, labels, color='red', label='Actual Salary')
plt.plot(features, model.predict(features), color='blue',
         label='Regression Line')

plt.title('Salary Prediction using Linear Regression')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Save the trained model
pickle.dump(model, open('Salary_model.pkl', 'wb'))

# Load the saved model
loaded_model = pickle.load(open('Salary_model.pkl', 'rb'))

# Predict salary for user input
years = float(input("Enter Years of Experience: "))
predicted_salary = loaded_model.predict(np.array([[years]]))

print("Estimated Salary for", years,
      "years of experience is:", predicted_salary[0])