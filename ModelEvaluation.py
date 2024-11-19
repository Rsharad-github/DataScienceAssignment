# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Sample dataset: Fraud detection example
data = pd.DataFrame({
    'Feature1': [1.2, 2.4, 3.1, 4.8, 5.5, 6.2, 7.1],
    'Feature2': [0.7, 1.5, 2.8, 3.6, 4.2, 5.3, 6.8],
    'Fraud': [0, 1, 0, 0, 1, 0, 1]  # Target variable: 1 indicates fraud
})

# Splitting the dataset
X = data[['Feature1', 'Feature2']]
y = data['Fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Training a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
# Making predictions
y_pred = model.predict(X_test)
# Basic confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
# Assigning unequal costs to false positives and false negatives
# Cost matrix: [ [TN, FP], [FN, TP] ]
cost_matrix = np.array([[0, 5],  # False Positive Cost = 5
                        [20, 0]])  # False Negative Cost = 20
# Calculating total cost
FP = conf_matrix[0][1]  # False Positives
FN = conf_matrix[1][0]  # False Negatives
total_cost = (FP * cost_matrix[0][1]) + (FN * cost_matrix[1][0])

print(f"Total Cost of Errors: {total_cost}")
# Adjusting the decision threshold
probs = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
threshold = 0.3  # Lower threshold to reduce False Negatives
# Custom predictions with adjusted threshold
y_pred_adjusted = (probs >= threshold).astype(int)
conf_matrix_adjusted = confusion_matrix(y_test, y_pred_adjusted)
print("Adjusted Confusion Matrix:\n", conf_matrix_adjusted)
# New cost with adjusted threshold
FP_adjusted = conf_matrix_adjusted[0][1]
FN_adjusted = conf_matrix_adjusted[1][0]
total_cost_adjusted = (FP_adjusted * cost_matrix[0][1]) + (FN_adjusted * cost_matrix[1][0])
print(f"Adjusted Total Cost of Errors: {total_cost_adjusted}")
# Classification report for adjusted threshold
print("\nClassification Report (Adjusted Threshold):\n", classification_report(y_test, y_pred_adjusted))

