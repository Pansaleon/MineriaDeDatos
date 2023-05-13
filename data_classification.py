import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar los datos
df = pd.read_csv('games.csv')

df['Number of Reviews'] = df['Number of Reviews'].str.replace('K', '').astype(float)
df['Wishlist'] = df['Wishlist'].str.replace('K', '').astype(float)
df['Plays'] = df['Plays'].str.replace('K', '').astype(float)
df['Plays'] = (df['Plays'] * 1000).astype(float)
df['Rating'] = (df['Rating']).astype(float)
df['Rating'] = df['Rating'].fillna('0').astype(float)

# Seleccionar las variables independientes y dependientes
X = df[['Wishlist']]  # Variable independiente
y = df['Number of Reviews']  # Variable dependiente

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las variables independientes
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear y entrenar el modelo de k-NN
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# Predecir los valores de 'Number of Reviews' en el conjunto de prueba
y_pred = knn.predict(X_test_scaled)

# Crear el plot
plt.scatter(X_test['Wishlist'], y_test, color='blue', label='Datos reales')
plt.scatter(X_test['Wishlist'], y_pred, color='red', label='Datos predichos')
plt.xlabel('Wishlist')
plt.ylabel('Number of Reviews')
plt.legend()
plt.show()