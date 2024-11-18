import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.DataFrame({
    'visits': [10, 20, 15, 25, 30, 12, 18],
    'ads_spent': [200, 400, 300, 500, 600, 250, 350],
    'discounts': [5, 10, 7, 12, 15, 6, 9]
})
print(data)

# Model fitting
poisson_model = smf.poisson(formula='visits ~ ads_spent + discounts', data=data).fit()

# Model summary
print(poisson_model.summary())

# New data
new_data = pd.DataFrame({
    'ads_spent': [250, 400, 550],
    'discounts': [8, 10, 12]
})

# Predictions
predicted_visits = poisson_model.predict(new_data)
print(predicted_visits)

# New data
new_data = pd.DataFrame({
    'ads_spent': [250, 400, 550],
    'discounts': [8, 10, 12]
})

import matplotlib.pyplot as plt

plt.scatter(data['ads_spent'], data['visits'], color='blue', label='Actual Data')
plt.plot(new_data['ads_spent'], predicted_visits, color='red', label='Predicted Data')
plt.xlabel('Ads Spent')
plt.ylabel('Visits')
plt.legend()
plt.show()

