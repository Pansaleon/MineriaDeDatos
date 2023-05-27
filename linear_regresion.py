import pandas as pd
import numpy as np
import ast
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
from dateutil.parser import parse
from wordcloud import WordCloud
from pandas.api.types import CategoricalDtype
# Machine Learning Model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def read_dataset():
    file_path = (r"D:\Docs\9no Semestre\Minería de Datos\MineriaDeDatos\games.csv")
    data = pd.read_csv(file_path, index_col = 0)
    return data

data = read_dataset()

#Visualizar datos

def scatter_plot(data, x_column, y_column):
    # Crear una figura y un conjunto de ejes
    fig, ax = plt.subplots()

    # Generar el gráfico de dispersión
    ax.scatter(data[x_column], data[y_column])

    # Configurar etiquetas y título del gráfico
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title('Gráfico de dispersión')
    # Utilizar la paleta de colores especificada en el gráfico de barras
    # Mostrar el gráfico
   


#
def calcular_puntuacion_popularidad(data):
    # Calcular la puntuación de popularidad para cada juego
    data['Popularidad'] = (data['Rating'] * 0.4) + (data['Number of Reviews'] * 0.3) + (data['Plays'] * 0.1) + (data['Wishlist'] * 0.2)
    
    # Ordenar los juegos por su puntuación de popularidad en orden descendente
    data = data.sort_values(by='Popularidad', ascending=False)
    
    return data

#Data transformation 

total_null = data.isnull().sum().sort_values(ascending = False)
percent = ((data.isnull().sum()/data.isnull().count())*100).sort_values(ascending = False)


missing_data = pd.concat([total_null,percent.round(2)],axis=1,keys=['Total Missing','In Percent'])




data['Rating'] = data['Rating'].replace(np.nan, 0.0)
data['Team'] = data['Team'].replace(np.nan, "['Unknown Team']")
data['Summary'] = data['Summary'].replace(np.nan, 'Unknown Summary')


total_null = data.isnull().sum().sort_values(ascending = False)
percent = ((data.isnull().sum()/data.isnull().count())*100).sort_values(ascending = False)

missing_data = pd.concat([total_null,percent.round(2)],axis=1,keys=['Total Missing','In Percent'])

data = data.drop_duplicates().sort_index()

data.loc[data['Release Date'] == 'releases on TBD']

#Convertir columna release_date a date
dt = datetime.now()

dt_str = dt.strftime('%b %d, %Y')

data['Release Date'] = data['Release Date'].str.replace('releases on TBD', dt_str )

data['Release Date'] = pd.to_datetime(data['Release Date'], format='%b %d, %Y')

data['Release Date'] = data['Release Date'].dt.strftime('%Y-%-m-%-d')

data['Release Date'] = pd.to_datetime(data['Release Date'])

# Obtener el día, mes y año de la fecha
data['Day'] = data['Release Date'].dt.day
data['Month'] = data['Release Date'].dt.strftime('%b')
data['Year'] = data['Release Date'].dt.year
data['Week day'] = data['Release Date'].dt.day_name()

data['Times Listed'] = data['Times Listed'].str.replace('K', '').astype(float) * 1000
data['Number of Reviews'] = data['Number of Reviews'].str.replace('K', '').astype(float) * 1000
data['Plays'] = data['Plays'].str.replace('K', '').astype(float) * 1000
data['Playing'] = data['Playing'].str.replace('K', '').astype(float) * 1000
data['Backlogs'] = data['Backlogs'].str.replace('K', '').astype(float) * 1000
data['Wishlist'] = data['Wishlist'].str.replace('K', '').astype(float) * 1000

data['Team'] = data['Team'].apply(lambda x: ast.literal_eval(x))

# create a sample DataFrame with a column containing multiple values
df_team = pd.DataFrame({
    'Title': data['Title'].tolist(),
    'Team': data['Team'].tolist()
})
# use the explode method to transform the 'Team' column
df_team = df_team.explode('Team')

data['Genres'] = data['Genres'].apply(lambda x: ast.literal_eval(x))

# create a sample DataFrame with a column containing multiple values
df_genres = pd.DataFrame({
    'Title': data['Title'].tolist(),
    'Genres': data['Genres'].tolist()
})
# use the explode method to transform the 'Team' column
df_genres = df_genres.explode('Genres')

data = data.drop(['Release Date', 'Team', 'Summary', 'Reviews'], axis = 1)


#Regresion lineal

def linear_regression(data, target_column, feature_columns):
    # Dividir los datos en variables dependientes (y) y variables independientes (X)
    X = data[feature_columns]
    y = data[target_column]

    # Crear una instancia del modelo de regresión lineal
    model = LinearRegression()

    # Ajustar el modelo a los datos
    model.fit(X, y)

    # Obtener los coeficientes de la regresión lineal
    coefficients = model.coef_
    intercept = model.intercept_

    return coefficients, intercept


# Especificar las columnas objetivo y las columnas de características para la regresión lineal
target_column = 'Rating'
feature_columns = ['Number of Reviews', 'Plays', 'Playing', 'Backlogs', 'Wishlist']

# Aplicar la función de regresión lineal
coefficients, intercept = linear_regression(data, target_column, feature_columns)

# Imprimir los coeficientes y la intersección de la regresión lineal
print('Coefficients:', coefficients)
print('Intercept:', intercept)




# Especificar las columnas x e y para el gráfico de dispersión
x_column = 'Number of Reviews'
y_column = 'Rating'

plt.title('Número de juegos lanzados por mes')

# Crear una lista con los nombres de los meses en orden
meses_ordenados = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Crear un objeto CategoricalDtype con el orden de los meses
meses_dtype = pd.CategoricalDtype(categories=meses_ordenados, ordered=True)

# Convertir la columna 'Month' al tipo de dato categórico con el orden de los meses
data['Month'] = data['Month'].astype(meses_dtype)

# Utilizar la paleta de colores especificada en el gráfico de barras
sns.countplot(data=data, y='Month', palette='coolwarm')

# Aplicar la función scatter_plot
scatter_plot(data, x_column, y_column)

#obtener el top 10 juegos del año

top_10_games = ['Fortnite', 'Minecraft', 'Grand Theft Auto V', 'Counter-Strike: Global Offensive', 'Apex Legends', 'League of Legends', 'Call of Duty: Warzone', 'Valorant', 'Dota 2', 'Roblox']
top_10_genres = ['Action', 'Shooter', 'Sports', 'Role-Playing', 'Adventure', 'Strategy', 'Simulation', 'Fighting', 'Racing', 'Puzzle']



top_rating = data[['Title','Rating']].sort_values(by = 'Rating', ascending = False)
top_rating = top_rating.loc[top_rating['Title'].isin(top_10_games)]
top_rating = top_rating.drop_duplicates()


fig, axes = plt.subplots(1, 2, figsize=(16, 5))

sns.histplot(ax = axes[0], data = data['Rating'], palette='coolwarm')
sns.barplot(ax = axes[1], data = top_rating, x = 'Rating', y = 'Title', palette = 'coolwarm')

axes[0].set_title('Distribución de la puntuación', pad = 5, fontsize = 13)
axes[0].set_xlabel('Rating', labelpad = 20)
axes[0].set_ylabel('Frequency', labelpad = 20)

axes[1].set_title('Top mejores juegos puntuados 2021', pad = 5, fontsize = 13)
axes[1].set_xlabel('Rating', labelpad = 20)
axes[1].set_ylabel('Title', labelpad = 20)
plt.tight_layout()



plt.figure(figsize=(16, 10))
plt.title('Numero de juegos lanzados por mes')
sns.countplot(data=data, y='Month')
sns.color_palette('coolwarm')



# Configurar el estilo de seaborn
sns.set()
#Calculando la popularidad de un juego por sus métricas
# Calcular la puntuación de popularidad para cada juego
data = calcular_puntuacion_popularidad(data)

# Obtener los 10 juegos más populares
top_10_populares = data.head(10)

# Crear el gráfico de barras
plt.figure(figsize=(12, 6))
plt.bar(top_10_populares['Title'], top_10_populares['Popularidad'], color='skyblue')

# Personalizar el gráfico
plt.title('Top 10 Juegos más Populares de Toda la Historia')
plt.xlabel('Juego')
plt.ylabel('Puntuación de Popularidad')
plt.xticks(rotation=90)

