# Importing necessary libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# Step 1: Creating the dataset
data = pd.DataFrame({
    'milk': [1, 0, 1, 1, 0],
    'bread': [1, 1, 1, 1, 1],
    'butter': [0, 1, 1, 0, 0],
    'cheese': [1, 0, 0, 1, 1]
})
print("Dataset:")
print(data)

# Step 2: Generating frequent itemsets
frequent_itemsets = apriori(data, min_support=0.5, use_colnames=True)
print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Step 3: Generating association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
print("\nAssociation Rules:")
print(rules)

# Step 4: Filtering strong rules based on lift and confidence
strong_rules = rules[(rules['lift'] > 1.2) & (rules['confidence'] > 0.7)]
print("\nStrong Rules (Lift > 1.2 and Confidence > 0.7):")
print(strong_rules)

# Step 5: Visualizing the results
plt.scatter(rules['support'], rules['confidence'], c=rules['lift'], cmap='viridis')
plt.colorbar(label='Lift')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Association Rules')
plt.show()
