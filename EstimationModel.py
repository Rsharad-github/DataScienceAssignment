import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("data.csv")#Comma-Separated Values
print(data.head())
# Name,Age,Score
# Alice,25,89
# Bob,30,92
# Charlie,22,85
#    Name  Age  Score
# 0  Alice   25     89
# 1    Bob   30     92
# 2 Charlie   22     85


X = data.drop("target_column", axis=1)  # Replace "target_column" with your actual target column name
y = data["target_column"]

#    feature1  feature2  target_column
# 0       1.0       2.0              0
# 1       3.0       4.0              1
# 2       5.0       6.0              0

# X:
#    feature1  feature2
# 0       1.0       2.0
# 1       3.0       4.0
# 2       5.0       6.0

# y:
# 0    0
# 1    1
# 2    0
# Name: target_column, dtype: int64




# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

# R-squared Score
r2 = r2_score(y_test, y_pred)
print("R-squared Score:", r2)

