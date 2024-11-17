# Step 1: Required Libraries import karte hain
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 2: Dummy Dataset create karte hain
data = {
    'Feature1': [2, 4, 6, 8, 10],
    'Feature2': [1, 3, 5, 7, 9],
    'Feature3': [10, 8, 6, 4, 2]
}
df = pd.DataFrame(data)

# Step 3: Data ko standardize karte hain
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Step 4: PCA apply karte hain
pca = PCA(n_components=2)  # Reduce dimensions to 2
pca_data = pca.fit_transform(scaled_data)

# Step 5: Explained variance ratio check karte hain
explained_variance = pca.explained_variance_ratio_

# Step 6: Visualize karte hain
plt.scatter(pca_data[:, 0], pca_data[:, 1], c='blue', marker='o')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection")
plt.grid()
plt.show()

# Step 7: Results print karte hain
print("Original Dataset:\n", df)
print("PCA Components:\n", pca.components_)
print("Explained Variance Ratio:\n", explained_variance)
