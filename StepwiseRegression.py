# Importing Libraries
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

# Load Boston Housing dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Forward Selection Function
def forward_selection(X, y):
    remaining_features = list(X.columns)
    selected_features = []
    best_model = None

    while remaining_features:
        results = {}
        for feature in remaining_features:
            temp_features = selected_features + [feature]
            X_temp = sm.add_constant(X[temp_features])  # Add constant for intercept
            model = sm.OLS(y, X_temp).fit()
            results[feature] = model.aic  # AIC for evaluation

        best_feature = min(results, key=results.get)
        if not best_model or results[best_feature] < best_model.aic:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            X_temp = sm.add_constant(X[selected_features])
            best_model = sm.OLS(y, X_temp).fit()
        else:
            break

    return best_model, selected_features

# Apply Forward Selection
model, selected_features = forward_selection(X_train, y_train)

print("Selected Features:", selected_features)
print("Model Summary:")
print(model.summary())

# Predict and Evaluate
X_test_selected = sm.add_constant(X_test[selected_features])
y_pred = model.predict(X_test_selected)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))



