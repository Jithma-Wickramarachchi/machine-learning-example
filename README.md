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