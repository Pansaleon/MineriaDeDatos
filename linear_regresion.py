import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# Load the dataset
df = pd.read_csv('games.csv')


# Replace 'k' and convert the column to float
df['Number of Reviews'] = df['Number of Reviews'].str.replace('K', '').astype(float)
df['Plays'] = df['Plays'].str.replace('K', '').astype(float)
df['Plays'] = (df['Plays'] * 1000).astype(float)
df['Rating'] = (df['Rating']).astype(float)
df['Rating'] = df['Rating'].fillna('0').astype(float)
# Split the dataset into independent variables (X) and the dependent variable (y)
X = df[['Plays']]  # Replace with the actual column names
y = df['Rating']  # Replace with the actual column name

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Predict the values of the dependent variable using the trained model
y_pred = model.predict(X)


# Plot the scatter plot of the data points
plt.scatter(X, y, color='blue', label='Actual')

# Plot the linear regression line
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')



# Set labels and title
plt.xlabel('Number of Reviews')
plt.ylabel('Rating')
plt.title('Line Graph')


# Display the graph
plt.show()