import pandas as pd
import numpy as np

df1 = pd.read_csv('../../data/England24_25.csv')
df2 = pd.read_csv('../../data/England25_26.csv')

df1['FTG'] = df1['FTHG'] + df1['FTAG'] # ilosc wszystkich goli
df2['FTG'] = df2['FTHG'] + df2['FTAG'] # ilosc wszystkich goli

columns_to_keep = ["Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR", # model target 1)
    "FTG", # model target 2)
    "HS",
    "AS",
    "HST",
    "AST",
    "HC",
    "AC",
    "HF",
    "AF",
    "HY",
    "AY",
    "HR",
    "AR",
    "B365H",
    "B365D",
    "B365A",
    "Avg>2.5",
    "Avg<2.5"
]

df1 = df1[columns_to_keep]
df2 = df2[columns_to_keep]