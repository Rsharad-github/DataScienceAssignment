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

# Step 4: Calculate Confidence Quotient
def confidence_quotient(row):
    return row['confidence'] / row['consequent support']

rules['confidence_quotient'] = rules.apply(confidence_quotient, axis=1)
print("\nRules with Confidence Quotient:")
print(rules[['antecedents', 'consequents', 'confidence', 'consequent support', 'confidence_quotient']])

# Step 5: Filter Rules Based on Confidence Quotient
significant_rules = rules[rules['confidence_quotient'] > 1.2]
print("\nSignificant Rules (Confidence Quotient > 1.2):")
print(significant_rules)

# Step 6: Visualize Confidence Quotient (Optional)
rules['confidence_quotient'].plot(kind='bar', title='Confidence Quotient of Rules')
plt.xlabel('Rules Index')
plt.ylabel('Confidence Quotient')
plt.show()
