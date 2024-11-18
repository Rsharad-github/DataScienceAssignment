# Step 1: Libraries import karte hain
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Dummy dataset create karte hain
data = {
    'Age': [22, 25, 47, 52, 46, 56, 28, 30, 22, 35],
    'Salary': [21000, 25000, 47000, 52000, 46000, 56000, 28000, 30000, 22000, 35000],
    'Purchased': [0, 0, 1, 1, 1, 1, 0, 0, 0, 1]
}
df = pd.DataFrame(data)

# Step 3: Features (X) aur Target (y) ko split karte hain
X = df[['Age', 'Salary']]
y = df['Purchased']

# Step 4: Train-Test split karte hain
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Logistic Regression Model fit karte hain
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Predictions karte hain
y_pred = model.predict(X_test)

# Step 7: Model evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Step 8: Confusion Matrix visualize karte hain
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 9: Results print karte hain
print("Accuracy Score:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
