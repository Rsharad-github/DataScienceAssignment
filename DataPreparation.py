
import pandas as pd
# Example dataset
data = pd.DataFrame({
    'Name': ['Aman', 'Bina', 'Charu'],
    'Age': [25, 30, 22]
})

# Adding an index field
data['Index'] = data.index + 1  # Starting index from 1
print(data)

# Change Misleading Field Values Using Python

data['Gender'] = ['M', 'F', 'Unknown']
data['Gender'] = data['Gender'].replace('Unknown', None)
print(data)

# Re Express Categorical Field Values Using Python
data['Gender'] = ['Male', 'FEMALE', 'male']
data['Gender'] = data['Gender'].str.lower().str.capitalize()
print(data)

# Standardise Numeric Fields Using Python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data['Age_Standardized'] = scaler.fit_transform(data[['Age']])
print(data)

# Identify Outliers Using Python
import numpy as np

Q1 = data['Age'].quantile(0.25)
Q3 = data['Age'].quantile(0.75)
IQR = Q3 - Q1
outliers = data[(data['Age'] < Q1 - 1.5 * IQR) | (data['Age'] > Q3 + 1.5 * IQR)]
print("Outliers:")
print(outliers)
