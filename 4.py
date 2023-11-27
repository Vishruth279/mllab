import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data = pd.read_csv('/content/Food-Truck-LineReg.csv', header=None, names=['X', 'Y'])

# Display the correlation matrix
correlation_matrix = data.corr()
print('Correlation Matrix:')
print(correlation_matrix)

# Extract features (X) and target variable (y)
X = data['X'].values.reshape(-1, 1)
y = data['Y'].values

# Create and fit the linear regression model
regression_model = LinearRegression()
regression_model.fit(X, y)

# Make predictions
y_pred = regression_model.predict(X)

# Calculate various metrics
cost = mean_squared_error(y, y_pred)
sse = np.sum((y - y_pred) ** 2)
ssr = np.sum((y_pred - np.mean(y)) ** 2)
sst = np.sum((y - np.mean(y)) ** 2)
r2 = r2_score(y, y_pred)

# Display metrics
print('Cost (Mean Squared Error):', cost)
print('SSE (Sum of Squared Errors):', sse)
print('SSR (Sum of Squared Regression):', ssr)
print('SST (Total Sum of Squares):', sst)
print('R-squared (R2):', r2)

# Display regression parameters
slope = regression_model.coef_[0]
intercept = regression_model.intercept_
print('Regression Parameters:')
print('Slope (Theta1):', slope)
print('Intercept (Theta0):', intercept)

# Plot the data and regression line
plt.scatter(X, y, label='Data points')
plt.plot(X, y_pred, color='red', linewidth=2, label='Linear Regression Line')
plt.title('Linear Regression Example')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
