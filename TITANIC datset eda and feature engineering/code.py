import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


titanic = pd.read_csv(r"C:\Users\Aditya\OneDrive\Desktop\ai ml\titanic.csv")
print(titanic.head())



print("\n--- DATA INFO ---")
print(titanic.info())

print("\n--- MISSING VALUES ---")
print(titanic.isnull().sum())

print("\n--- SUMMARY STATISTICS ---")
print(titanic.describe())

sns.countplot(x='Survived', data=titanic)
plt.title("Survival Count (0 = Died, 1 = Survived)")
plt.show()

sns.histplot(titanic['Age'], kde=True)
plt.title("Age Distribution")
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=titanic)
plt.title("Survival by Passenger Class")
plt.show()

sns.countplot(x='Sex', hue='Survived', data=titanic)
plt.title("Survival by Sex")
plt.show()

sns.countplot(x='Embarked', hue='Survived', data=titanic)
plt.title("Survival by Port of Embarkation")
plt.show()


print("\n--- DATA CLEANING ---")
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)

titanic['Fare'].fillna(titanic['Fare'].median(), inplace=True)

titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)

titanic.drop(['PassengerId'], axis=1, inplace=True)

print("Missing values after cleaning:")
print(titanic.isnull().sum())


print("\n--- FEATURE ENGINEERING ---")
titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch'] + 1

titanic = pd.get_dummies(titanic, columns=['Sex', 'Embarked'], drop_first=True)

print("New columns after encoding:")
print(titanic.columns)


print("\n--- DATA PREPARATION ---")
X = titanic.drop('Survived', axis=1)
y = titanic['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

