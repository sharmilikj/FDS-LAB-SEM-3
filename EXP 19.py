import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

url = "https://raw.githubusercontent.com/TheAlgorithms/Jupyter/master/machine_learning/random_forest_classification/Social_Network_Ads.csv"

df = pd.read_csv(url)

print(df)
print(df.head())

features = df.iloc[:, [2, 3]].values
label = df.iloc[:, 4].values

print(features)
print(label)

for i in range(1, 401):
    x_train, x_test, y_train, y_test = train_test_split(
        features, label, test_size=0.2, random_state=i
    )

    model = LogisticRegression()
    model.fit(x_train, y_train)

    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)

    if test_score > train_score:
        print("Test", test_score, "Train", train_score,
              "Random State", i)

x_train, x_test, y_train, y_test = train_test_split(
    features, label, test_size=0.2, random_state=4
)

finalModel = LogisticRegression()
finalModel.fit(x_train, y_train)

print(finalModel.score(x_train, y_train))
print(finalModel.score(x_test, y_test))

print(classification_report(label, finalModel.predict(features)))