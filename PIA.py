import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Cargar el conjunto de datos desde el archivo CSV
data = pd.read_csv('boston.csv')

# Dividir los datos en características (X) y etiquetas (y)
X = data.drop('MEDV', axis=1)
y = data['MEDV']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el objeto de regresión lineal
regression = LinearRegression()

# Entrenar el modelo utilizando los datos de entrenamiento
regression.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = regression.predict(X_test)

# Calcular el error cuadrado medio (MSE) en el conjunto de prueba
mse = np.mean((y_pred - y_test) ** 2)
print("Mean Squared Error (MSE):", mse)

# Graficar las predicciones y los valores reales
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Valores reales')
plt.ylabel('Predicciones')
plt.title('Regresión lineal: Valores reales vs. Predicciones')
plt.show()
