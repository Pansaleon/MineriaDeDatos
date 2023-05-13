import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def forecast(model, X):
    # Predict the values of the dependent variable using the trained model
    y_pred = model.predict(X)
    return y_pred


# Load the dataset
df = pd.read_csv('games.csv')

# Replace 'k' and convert the columns to float
df['Number of Reviews'] = df['Number of Reviews'].str.replace('K', '').astype(float)
df['Plays'] = df['Plays'].str.replace('K', '').astype(float)
df['Rating'] = df['Rating'].fillna('0').astype(float)
# Split the dataset into independent variables (X) and the dependent variable (y)
X = df[['Plays']]  # Replace with the actual column names
y = df['Rating']  # Replace with the actual column name

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Call the forecast function to predict new values
X_new = [[1100], [1150], [1200], [1250], [1300]]  # Replace with the new data points you want to forecast
y_pred = forecast(model, X_new)

# Print the predicted values
print(y_pred)

# Plot the scatter plot of the data points
plt.scatter(X, y, color='blue', label='Actual')

# Plot the linear regression line
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')

# Plot the forecasted values
plt.scatter(X_new, y_pred, color='green', label='Forecast')

# Set labels and title
plt.xlabel('Number of Reviews')
plt.ylabel('Rating')
plt.title('Line Graph')

# Display the graph
plt.legend()
plt.show()
