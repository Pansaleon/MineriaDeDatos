import requests
import io
from bs4 import BeautifulSoup
import pandas as pd
from tabulate import tabulate
from typing import Tuple, List
import re
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('games.csv')

# Crear una lista vacía
my_list = []
separator = ' '

# Iterar sobre cada fila del DataFrame
for index, row in df.iterrows():
    genres = row['Genres']
    # Eliminar las llaves del string
    genres = genres.strip('[]')
    words = re.findall(r'\b\w+\b', genres)
    for word in words:
        my_list.append(word)

# Join the list elements into a string
my_string = separator.join(my_list)


# Abrir el archivo en modo lectura
file = open('archivo.txt', 'r')

file.write(my_string)
# Leer el contenido del archivo
content = file.read()

# Cerrar el archivo
file.close()

# Imprimir el contenido del archivo
print(content)

    
columnn_genres = df['Genres']
# String con palabras separadas por comas y entre llaves
string = "['Adventure', 'Brawler', 'RPG']"

# Eliminar las llaves del string
string = string.strip('[]')

# Dividir el string en palabras utilizando una expresión regular y la coma como delimitador
words = re.findall(r'\b\w+\b', string)

# Imprimir el array de palabras
print(my_list)