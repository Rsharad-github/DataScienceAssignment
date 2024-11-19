# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# Sample dataset
data = pd.DataFrame({
    'Feature1': [5, 7, 8, 6, 9],
    'Feature2': [2, 4, 5, 7, 8],
    'Target': [1, 0, 0, 0, 1]
})
# Splitting the dataset into features (X) and target (y)
X = data[['Feature1', 'Feature2']]  # Independent variables
y = data['Target']  # Dependent variable
# Partitioning the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Splitting the majority and minority classes
majority = data[data['Target'] == 0]
minority = data[data['Target'] == 1]
# Upsampling the minority class
minority_upsampled = resample(minority, 
                              replace=True,    # Sample with replacement
                              n_samples=len(majority),  # Match majority class size
                              random_state=42)
# Combining the balanced classes
balanced_data = pd.concat([majority, minority_upsampled])


# Building the CART Decision Tree
cart_model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
cart_model.fit(X_train, y_train)
# Visualizing the Decision Tree
plt.figure(figsize=(10, 8))
plot_tree(cart_model, feature_names=X.columns, class_names=['Not Survived', 'Survived'], filled=True)
plt.title("CART Decision Tree")
plt.show()
# Printing the textual representation of the tree
tree_rules = export_text(cart_model, feature_names=list(X.columns))
print("Decision Tree Rules:")
print(tree_rules)
# Evaluating the model
accuracy = cart_model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


# Building the C5.0 Decision Tree (sklearn uses CART; similar logic applies)
c5_model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)  # Using 'entropy' for information gain
c5_model.fit(X_train, y_train)
# Visualizing the Decision Tree
plt.figure(figsize=(10, 8))
plot_tree(c5_model, feature_names=X.columns, class_names=['Not Survived', 'Survived'], filled=True)
plt.title("C5.0 Decision Tree Visualization")
plt.show()
# Printing tree rules
tree_rules = export_text(c5_model, feature_names=list(X.columns))
print("C5.0 Decision Tree Rules:")
print(tree_rules)
# Evaluating the model
accuracy = c5_model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


# Building the Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)
# Making predictions
y_pred = rf_model.predict(X_test)
# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# Feature importance
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
# Visualizing feature importance
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title("Feature Importance in Random Forest")
plt.show()
