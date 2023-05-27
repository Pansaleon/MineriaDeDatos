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
from collections import Counter

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
    plt.show()


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
df_team = df_team.explode('Team')

print(df_team.Team)

nombres_genres = " ".join(df_team['Team'].tolist())

with open('Team.txt', 'w', encoding='utf-8') as f:
    f.write(nombres_genres)

def open_file(path: str) -> str:
    content = ""
    with open(path, "r") as f:
        content = f.readlines()
    return " ".join(content)


all_words = ""
frase = open_file("Team.txt") # "hola a todos muchas  palabras palabras hola muchas hola hola hola palabras palabras hola muchas hola hola hola palabras palabras hola muchas hola hola hola palabras palabras hola muchas hola hola hola"
palabras = frase.rstrip().split(" ")

Counter(" ".join(palabras).split()).most_common(10)
# looping through all incidents and joining them to one text, to extract most common words
for arg in palabras:
    tokens = arg.split()
    all_words += " ".join(tokens) + " "

print(all_words)
wordcloud = WordCloud(
    background_color="white", min_font_size=5
).generate(all_words)

# print(all_words)
# plot the WordCloud image
plt.close()
plt.figure(figsize=(5, 5), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

# plt.show()
plt.savefig("img/word_cloud.png")
plt.close()