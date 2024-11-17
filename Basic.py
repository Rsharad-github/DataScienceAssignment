# Ye ek single-line comment hai
print("Hello, Python!")  # Iske baad bhi comment likh sakte hain

# Executing Commands in Python
# python your_script.py

# Importing Packages in R and Python
import numpy as np
import pandas as pd

# Getting Data into python
data = pd.read_csv("data.csv") #1,2 ,3
print(data.head())

# Saving Output in Python
data.to_csv("output.csv", index=False)
print("Data saved in Python!")

# Accessing Records and Variables in  Python
print(data['column_name'])  # Ek column access karna
print(data.iloc[0])  # Pehla row access karna
