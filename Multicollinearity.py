# Step 1: Necessary Libraries Import karte hain
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Step 2: Sample dataset banate hain
data = {
    'Feature1': [10, 20, 30, 40, 50],
    'Feature2': [15, 25, 35, 45, 55],
    'Feature3': [100, 200, 300, 400, 500],
    'Feature4': [1, 2, 3, 4, 5]
}
df = pd.DataFrame(data)

# Step 3: Correlation Matrix calculate karte hain
correlation_matrix = df.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Step 4: Correlation Heatmap plot karte hain
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Step 5: Variance Inflation Factor (VIF) calculate karte hain
def calculate_vif(dataframe):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = dataframe.columns
    vif_data["VIF"] = [
        variance_inflation_factor(dataframe.values, i)
        for i in range(dataframe.shape[1])
    ]
    return vif_data

vif = calculate_vif(df)
print("Variance Inflation Factor (VIF):")
print(vif)
