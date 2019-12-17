import pandas as pd
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

df1 = pd.read_csv("articles1.csv", usecols=[2,3,4,5,9])
df2 = pd.read_csv("articles2.csv", usecols=[2,3,4,5,9])
df3 = pd.read_csv("articles3.csv", usecols=[2,3,4,5,9])

df = df1.append(df2, ignore_index=True, sort=False).append(df3, ignore_index=True, sort=False)
df = df[df['publication'].isin(pubs.keys())]
