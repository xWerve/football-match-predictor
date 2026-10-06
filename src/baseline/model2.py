from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from import_data import df1, df2

train = df1
test = df2

# Przygotowujemy zbiór treningowy cech X i celów y
# Cechy to na razie same nazwy drużyn (HomeTeam i AwayTeam)
X_train = train[["HomeTeam", "AwayTeam"]]
y_train_home = train["FTHG"]
y_train_away = train["FTAG"]

# Tworzymy 2 modele: jeden dla goli gospodarza, drugi dla goli gościa
pipe_home = Pipeline(
    [
        (
            "ohe",
            OneHotEncoder(handle_unknown="ignore"),
        ),  # radzi sobie z nowymi drużynami
        ("model", PoissonRegressor(alpha=1e-3)),
    ]
)

pipe_away = Pipeline(
    [
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ("model", PoissonRegressor(alpha=1e-3)),
    ]
)

# Trenujemy
pipe_home.fit(X_train, y_train_home)
pipe_away.fit(X_train, y_train_away)

# Predykcja na zbiorze testowym:
X_test = test[["HomeTeam", "AwayTeam"]]
test["pred_lambda_home"] = pipe_home.predict(X_test)
test["pred_lambda_away"] = pipe_away.predict(X_test)
print(test[["HomeTeam", "AwayTeam", "pred_lambda_home", "pred_lambda_away"]].head())