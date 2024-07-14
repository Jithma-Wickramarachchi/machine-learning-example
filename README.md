# machine-learning-example
This is a codebase which I use to practice ML concepts

## Pandas
### a python library which use to clean and analyse big data.

```python
import pandas as pd

# create DataFrame
df = pd.read_csv('vgsales.csv')

# view (rows, columns)
df.shape

# describe dataframe
df.describe()

# Create another dataframe
# Without the column
# Which we want to predict
x = df.drop(columns='Global_Sales')

# Create another df
# With the column
# Which we want to predict
y = df['Global_Sales']
```

## Scikit-learn
### A library to create ml models

```python
from sklearn.tree import DecisionTreeClassifier

# Create a model according to the algorithm
model = DecisionTreeClassifier()

# Set features and labels
model.fit(X, y)

# Predict specific labels
# When features like in array
predictions = model.predict([[12, 14], [45, 47]])

# Print predicted results
predictions
```
### Split dataset for training and testing

```python
# Split data into training and testing groups
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# Define the percentage of testing size
# In this case, 20%
# This returns tuple
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
# Set data to train
model.fit(X_train, y_train)
# Set data to predict
predictions = model.predict(X_test)
```
### Check accuracy of the Model

```python
from sklearn.metrics import accuracy_score
# Compare predicted results with actual results
score = accuracy_score(y_test, predictions)

```