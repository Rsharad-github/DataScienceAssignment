import matplotlib.pyplot as plt
import numpy as np

# Sample data
categories = ['A', 'B', 'C', 'D']
values1 = [5, 7, 8, 6]  # Dataset 1
values2 = [6, 5, 7, 8]  # Dataset 2

# Creating bar graph with overlay
x = np.arange(len(categories))  # Positions for bars
width = 0.4  # Width of bars

plt.bar(x - width/2, values1, width, label='Dataset 1', color='blue')  # First bar graph
plt.bar(x + width/2, values2, width, label='Dataset 2', color='orange')  # Second bar graph

plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Graph with Overlay')
plt.xticks(x, categories)  # Adding category labels
plt.legend()
plt.show()
import pandas as pd

# Example dataset
data = pd.DataFrame({
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Female'],
    'Purchased': ['Yes', 'No', 'Yes', 'No', 'Yes']
})

# Constructing contingency table
contingency_table = pd.crosstab(data['Gender'], data['Purchased'])
print("Contingency Table:")
print(contingency_table)
# Sample data
data1 = np.random.normal(50, 10, 1000)  # First dataset (mean=50, std=10)
data2 = np.random.normal(60, 15, 1000)  # Second dataset (mean=60, std=15)

# Plotting histogram with overlay
plt.hist(data1, bins=30, alpha=0.5, label='Dataset 1', color='blue')  # Histogram 1
plt.hist(data2, bins=30, alpha=0.5, label='Dataset 2', color='orange')  # Histogram 2

plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram with Overlay')
plt.legend()
plt.show()

# Example dataset
data = pd.DataFrame({
    'Age': [22, 25, 27, 35, 45, 52, 60],
    'Purchased': [0, 0, 1, 1, 0, 1, 1]
})

# Defining bins and labels
bins = [0, 30, 50, 70]  # Age intervals
labels = ['Young', 'Middle-aged', 'Senior']

# Adding binned column
data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)
print("Dataset with Binned Age Groups:")
print(data)
