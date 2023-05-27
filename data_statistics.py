import pandas as pd
import numpy as np 

# Lee el archivo CSV y crea un dataframe
df = pd.read_csv(r"D:\Docs\9no Semestre\Minería de Datos\MineriaDeDatos\games.csv")


def count_unique_values(df, column_name):
    return df[column_name].nunique()





#Obtiene los Los  juegos con mejores calificacion del año indicado
def get_top_games_by_year(df, release_date_col, rating_col):
    # Convertir la columna de fecha de lanzamiento a tipo fecha
    df[release_date_col] = pd.to_datetime(df[release_date_col], errors='coerce')

    # Descartar las filas que contienen la cadena "TBD" en la columna de fecha de lanzamiento
    df = df[~df[release_date_col].astype(str).str.contains('TBD')]

    # Agrupar los juegos por año y encontrar los índices de las filas con las calificaciones más altas
    idx = df.groupby(df[release_date_col].dt.year)[rating_col].idxmax()

    # Eliminar los índices que contienen valores NaN de la lista de índices
    idx = idx[~np.isnan(idx)]

    # Seleccionar las filas correspondientes del dataframe original
    top_games_by_year = df.loc[idx][[release_date_col, 'Title', rating_col]]

    return top_games_by_year


# Muestra el dataframe
print(df.head())

# Cuenta el número de datos distintos dentro de la columna Team
num_unique_teams = count_unique_values(df, 'Team')
print('Número de equipos distintos:', num_unique_teams)

#Muestra los top 10 juegos del todos los años
top_games = get_top_games_by_year(df, 'Release Date', 'Rating')
print(top_games)



