import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import ast
pd.plotting.register_matplotlib_converters()
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



def read_dataset():
    file_path = (r"D:\Docs\9no Semestre\Minería de Datos\MineriaDeDatos\games.csv")
    data = pd.read_csv(file_path, index_col = 0)
    return data

def calcular_puntuacion_popularidad(data):
    # Calcular la puntuación de popularidad para cada juego
    data['Popularidad'] = (data['Rating'] * 0.4) + (data['Number of Reviews'] * 0.3) + (data['Plays'] * 0.1) + (data['Wishlist'] * 0.2)
    
    # Ordenar los juegos por su puntuación de popularidad en orden descendente
    data = data.sort_values(by='Popularidad', ascending=False)
    
    return data

data = read_dataset()



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



#Tratar la data duplicada



data = data.drop_duplicates().sort_index()

#Ya no tenemos data duplicada


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





# Supongamos que tienes un DataFrame llamado 'data' que contiene tus datos
# y quieres predecir el nivel de popularidad ('Popularity') de un juego
data = calcular_puntuacion_popularidad(data)

# Set the size of the plot
plt.figure(figsize=(12, 8))

# Set the style of the plot
sns.set_style('whitegrid')

# Plot the joint plot
sns.jointplot(
    x='Rating',
    y='Number of Reviews',
    data=data,
    alpha=0.8,
    edgecolor='black',
    linewidth=0.5,
    s=80,
    kind='scatter',
)

# Set the axis labels and title
plt.xlabel('Rating', fontsize=14)
plt.ylabel('Number of Reviews', fontsize=14)
plt.title('Relación entre el rating y el número de reviews', fontsize=16, pad=100)

# Set the axis tick label size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add grid lines to the plot
plt.grid(True)

# Show the plot
plt.show()

# Plot the joint plot with regression line
sns.regplot(
    x='Rating',
    y='Number of Reviews',
    data=data,
    scatter_kws={'alpha': 0.8, 'edgecolor': 'black', 'linewidth': 0.5, 's': 80},
    line_kws={'color': 'red'},
)

# Set the axis labels and title
plt.xlabel('Rating', fontsize=14)
plt.ylabel('Number of Reviews', fontsize=14)
plt.title('Relación entre el rating y el número de reviews', fontsize=16, pad=100)

# Set the axis tick label size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add grid lines to the plot
plt.grid(True)

# Show the plot
plt.show()
