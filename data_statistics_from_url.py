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



def read_dataset():
    file_path = (r"D:\Docs\9no Semestre\Minería de Datos\MineriaDeDatos\games.csv")
    data = pd.read_csv(file_path, index_col = 0)
    return data

data = read_dataset()

print(data.head())

print(data.info())

print(data.nunique()) 

#Data transformation 

total_null = data.isnull().sum().sort_values(ascending = False)
percent = ((data.isnull().sum()/data.isnull().count())*100).sort_values(ascending = False)
print("Total records = ", data.shape[0])

missing_data = pd.concat([total_null,percent.round(2)],axis=1,keys=['Total Missing','In Percent'])
print(missing_data)



data['Rating'] = data['Rating'].replace(np.nan, 0.0)
data['Team'] = data['Team'].replace(np.nan, "['Unknown Team']")
data['Summary'] = data['Summary'].replace(np.nan, 'Unknown Summary')


total_null = data.isnull().sum().sort_values(ascending = False)
percent = ((data.isnull().sum()/data.isnull().count())*100).sort_values(ascending = False)
print("Total records = ", data.shape[0])

missing_data = pd.concat([total_null,percent.round(2)],axis=1,keys=['Total Missing','In Percent'])

print(missing_data)

#Tratar la data duplicada

print(data[data.duplicated()])

data = data.drop_duplicates().sort_index()

#Ya no tenemos data duplicada
print(data.duplicated().any())

data.loc[data['Release Date'] == 'releases on TBD']

#Convertir columna release_date a date
dt = datetime.now()

dt_str = dt.strftime('%b %d, %Y')
print(dt_str)


data['Release Date'] = data['Release Date'].str.replace('releases on TBD', dt_str )

data['Release Date'] = pd.to_datetime(data['Release Date'], format='%b %d, %Y')

data['Release Date'] = data['Release Date'].dt.strftime('%Y-%-m-%-d')

data['Release Date'] = pd.to_datetime(data['Release Date'])

# Obtener el día, mes y año de la fecha
data['Day'] = data['Release Date'].dt.day
data['Month'] = data['Release Date'].dt.strftime('%b')
data['Year'] = data['Release Date'].dt.year
data['Week day'] = data['Release Date'].dt.day_name()


print(data[['Release Date', 'Day', 'Month', 'Year', 'Week day']].head())



data['Times Listed'] = data['Times Listed'].str.replace('K', '').astype(float) * 1000
data['Number of Reviews'] = data['Number of Reviews'].str.replace('K', '').astype(float) * 1000
data['Plays'] = data['Plays'].str.replace('K', '').astype(float) * 1000
data['Playing'] = data['Playing'].str.replace('K', '').astype(float) * 1000
data['Backlogs'] = data['Backlogs'].str.replace('K', '').astype(float) * 1000
data['Wishlist'] = data['Wishlist'].str.replace('K', '').astype(float) * 1000



print(data.describe())


data['Team'] = data['Team'].apply(lambda x: ast.literal_eval(x))

# create a sample DataFrame with a column containing multiple values
df_team = pd.DataFrame({
    'Title': data['Title'].tolist(),
    'Team': data['Team'].tolist()
})
# use the explode method to transform the 'Team' column
df_team = df_team.explode('Team')
print(df_team)



data['Genres'] = data['Genres'].apply(lambda x: ast.literal_eval(x))

# create a sample DataFrame with a column containing multiple values
df_genres = pd.DataFrame({
    'Title': data['Title'].tolist(),
    'Genres': data['Genres'].tolist()
})
# use the explode method to transform the 'Team' column
df_genres = df_genres.explode('Genres')
print(df_genres)


data = data.drop(['Release Date', 'Team', 'Summary', 'Reviews'], axis = 1)

print(data.head())



