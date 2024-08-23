import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'Mobile-Price-Prediction-cleaned_data.csv'
data = pd.read_csv(file_path)

# Define the features (X) and the target (y)
X = data.drop('Price', axis=1)
y = data['Price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate accuracy metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print the accuracy metrics
print(f"R² Score: {r2}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
plt.show()

# Print the equation of the regression line
intercept = model.intercept_
coefficients = model.coef_
equation = f"y = {intercept:.2f}"
for i, coef in enumerate(coefficients):
    equation += f" + ({coef:.2f} * X{i+1})"
print("Equation of the regression line:")
print(equation)

# Predict some sample values
sample_values = X_test.head(5)
sample_predictions = model.predict(sample_values)
print("\nSample predictions:")
print(sample_predictions)

# Display the sample values and their predictions
sample_values_df = pd.DataFrame(sample_values)
sample_values_df['Predicted Price'] = sample_predictions
print("\nSample values with predicted prices:")
print(sample_values_df)
