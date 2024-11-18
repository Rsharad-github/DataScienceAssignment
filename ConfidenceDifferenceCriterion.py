# Import Required Libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# Step 1: Example Dataset
data = {
    'Milk': [1, 0, 1, 1, 0],
    'Bread': [1, 1, 1, 0, 1],
    'Butter': [0, 1, 1, 1, 0],
    'Cheese': [0, 0, 1, 1, 1]
}
df = pd.DataFrame(data)
print("Dataset:")
print(df)

# Step 2: Generate Frequent Itemsets Using Apriori
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Step 3: Generate Association Rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print("\nAssociation Rules:")
print(rules)

# Step 4: Calculate Confidence Difference
def confidence_difference(row):
    return abs(row['confidence'] - row['consequent support'])

rules['confidence_diff'] = rules.apply(confidence_difference, axis=1)
print("\nRules with Confidence Difference:")
print(rules[['antecedents', 'consequents', 'confidence', 'consequent support', 'confidence_diff']])

# Step 5: Filter Significant Rules Based on Confidence Difference
significant_diff_rules = rules[rules['confidence_diff'] > 0.2]
print("\nSignificant Rules (Confidence Difference > 0.2):")
print(significant_diff_rules)

# Step 6: Visualize Confidence Difference (Optional)
rules['confidence_diff'].plot(kind='bar', title='Confidence Difference of Rules')
plt.xlabel('Rules Index')
plt.ylabel('Confidence Difference')
plt.show()
