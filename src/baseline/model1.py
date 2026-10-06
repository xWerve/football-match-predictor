import numpy as np
import pandas as pd
from scipy.stats import poisson
import statsmodels.api as sm
import statsmodels.formula.api as smf
from import_data import *

# 1. Łączymy dane (np. df1 jako trening, df2 jako test)
train = df1.copy()
test = df2.copy()

# 2. Przekształcamy dane do formatu: [team, opponent, home, goals]
home_df = train[["HomeTeam", "AwayTeam", "FTHG"]].rename(
    columns={"HomeTeam": "team", "AwayTeam": "opponent", "FTHG": "goals"}
)
home_df["home"] = 1

away_df = train[["AwayTeam", "HomeTeam", "FTAG"]].rename(
    columns={"AwayTeam": "team", "HomeTeam": "opponent", "FTAG": "goals"}
)
away_df["home"] = 0

df_glm = pd.concat([home_df, away_df])

# 3. Trenujemy klasyczny model Poissona (GLM)
model = smf.glm(
    formula="goals ~ home + team + opponent",
    data=df_glm,
    family=sm.families.Poisson(),
).fit()

# print(model.summary())  # Zobaczysz siły ataku i obrony poszczególnych klubów!


# 4. Funkcja do predykcji oczekiwanych goli (lambd) dla konkretnego meczu:
def predict_lambdas(home_team, away_team):
    lh = model.predict(
        pd.DataFrame([{"team": home_team, "opponent": away_team, "home": 1}])
    ).values[0]
    la = model.predict(
        pd.DataFrame([{"team": away_team, "opponent": home_team, "home": 0}])
    ).values[0]
    return lh, la


# Test dla meczu z df2:
lh, la = predict_lambdas("Crystal Palace", "Wolves")
print(f"Oczekiwane gole: Crystal Palace={lh:.2f}, Wolves={la:.2f}")
print(f"Oczekiwane gole: Crystal Palace={lh:.0f}, Wolves={la:.0f}")