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
df = pd.read_csv(r"C:\Users\Dell\Desktop\mineria-datos-rsjm\MineriaDeDatos\India Tourism 2014-2020\Country Quater Wise Visitors.csv")
df2 = pd.read_csv(r"C:\Users\Dell\Desktop\mineria-datos-rsjm\MineriaDeDatos\India Tourism 2014-2020\General Data 2014-2020.csv")
df3 = pd.read_csv(r"C:\Users\Dell\Desktop\mineria-datos-rsjm\MineriaDeDatos\India Tourism 2014-2020\Country Wise Age Group.csv")
print_tabulate(df)
print_tabulate(df2)
