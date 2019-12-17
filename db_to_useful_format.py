import pandas as pd
import sqlite3

pd.set_option('display.max_columns', 7)
pd.set_option('display.width', 200)

#These are the news websites and their respective bias scores.
#-2 means most liberal, +2 means most conservative
pubs = {
    "Atlantic": -1,
    "Breitbart": 2,
    "Buzzfeed News": -1,
    "CNN": -1,
    "Fox News": 1,
    "Guardian": -1,
    "National Review": 2,
    "New York Post": 2,
    "New York Times": -1,
    "NPR": 0,
    "Reuters": 0,
    "Vox": -2,
    "Washington Post": -1}

cnx = sqlite3.connect('all-the-news.db')
df = pd.read_sql_query("SELECT title, author, date, content, publication FROM longform", cnx)
df = df[df['publication'].isin(pubs.keys())]
print(df)