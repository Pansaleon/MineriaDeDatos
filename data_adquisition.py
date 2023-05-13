#Function declaration

import requests
import io
from bs4 import BeautifulSoup
import pandas as pd
from tabulate import tabulate
from typing import Tuple, List
import re
from datetime import datetime

def get_soup(url: str) -> BeautifulSoup:
    #implementar tr/catch por si la página está caída
    response = requests.get(url)
    return BeautifulSoup(response.content, 'html.parser')

def get_csv_from_url(url:str) -> pd.DataFrame:
    s=requests.get(url).content
    return pd.read_csv(io.StringIO(s.decode('utf-8')))

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='orgtbl'))

#Code

#from file
df = pd.read_csv(r"D:\Docs\9no Semestre\Minería de Datos\MineriaDeDatos\games.csv")

print_tabulate(df)

df.columns = ['index',
       'Titulo',
       'Fecha de Salida', 'Desarrolladores', 'Calificacion', 'Veces listado',
       'Numero de reviews', 'Generos',
       'Comentarios']

print(df.columns)

# print(df.columns)