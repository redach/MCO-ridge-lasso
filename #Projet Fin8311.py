#Projet Fin8311

import yfinance as yf
import pandas as pd
import numpy as np

# 15 titres du portefeuille
tickers = ["WCP.TO", "CVE.TO", "SU.TO", "AYA.TO", "BTO.TO", "DPM.TO", "SLF.TO", "BMO.TO", 
           "QBR-B.TO", "ACO-X.TO", "FTS.TO", "EXE.TO", "SIA.TO", "CLS.TO", "SHOP.TO"]

#Prix mensuel depuis 2005
donnees = yf.download(tickers, start="2005-01-01", interval="1mo", auto_adjust=True)
prix = donnees["Close"]
debut = prix.apply(lambda col: col.first_valid_index())
print(debut.sort_values())
print(prix.shape)

# Arreter l'échantillon à octobre 2026
prix2 = prix.loc[:"2026-10"]
#garder mois ou tous les titres ont un prix
prix = prix2.dropna()
#rendements mensuels
rendements = prix2.pct_change().dropna()
print(rendements.index[0], rendements.index[-1])
print(rendements.shape)

#poids en %
poids_pct = pd.Series({"WCP.TO":8 , "CVE.TO": 8, "SU.TO": 8, "AYA.TO": 8, "BTO.TO" : 7, "DPM.TO": 8, 
                       "SLF.TO": 6, "BMO.TO": 6, "QBR-B.TO": 6, "ACO-X.TO": 6, "FTS.TO": 7, "EXE.TO": 5, "SIA.TO": 5, "CLS.TO": 6, "SHOP.TO": 6})
print(poids_pct.sum())
poids = poids_pct / 100
rend_ptf = rendements.dot(poids)
print(rend_ptf.head())
print(rend_ptf.describe())

#Taux de change mensuel cad us
fx = yf.download("CADUSD=X", start="2015-05-01", end="2026-11-01",
                 interval="1mo", auto_adjust=True)["Close"].squeeze()
rend_fx = fx.pct_change().dropna()
#rendement du portefeuille
rend_ptf_usd = (1 + rend_ptf) * (1 + rend_fx) - 1

#Vérifications
print(rend_ptf_usd.shape, rend_ptf_usd.isna().sum())
print(rend_ptf_usd.describe().round(4))

import pandas_datareader.data as web
#1. Télécharger les facteur fama french NA
ff5 = web.DataReader("North_America_5_Factors", "famafrench", start="2015-06")[0]
mom = web.DataReader("North_America_Mom_Factor", "famafrench", start="2015-06")[0]

#Réeunir les deux tableaux 
facteurs = ff5.join(mom)
facteurs.columns = facteurs.columns.str.strip()
facteurs = facteurs.rename(columns={"Mkt-RF": "MKT", "WML": "MOM"})
facteurs = facteurs / 100

#Vérification
print(facteurs.index[0], facteurs.index[-1])
print(facteurs.shape)
print(facteurs.tail(3).round(4))

# ajout du pétrole et de l'or comme facteurs
mp_j = yf.download(["CL=F", "GC=F"], start="2015-05-01", end="2026-11-01", 
                auto_adjust=True)["Close"]
mp_j = mp_j.rename(columns={"CL=F": "PETROLE", "GC=F": "OR"})

#rendements mensuels
mp = mp_j.groupby(mp_j.index.to_period("M")).last()
rend_mp = mp.pct_change().dropna()
variables = facteurs.join(rend_mp, how="inner")
#Vérifications
print(mp.shape)
print(variables.shape, variables.isna().sum().sum())
print(rend_mp.agg(["min", "max"]).round(3))

#Mettre les dates du portefeuille en format mensuel
ptf = rend_ptf_usd.copy()
ptf.index = ptf.index.to_period("M")
ptf.name = "PTF"
#Reunir portefeuille et variables explicatives
data = variables.join(ptf, how="inner")
#Excès de rendement du portefeuille
data["EXCES"] = data["PTF"] - data["RF"]

#Vérification
print(data.shape, data.isna().sum().sum())
print(data.index[0], data.index[-1])
print(data[["PTF", "RF", "EXCES"]].describe().round(4))

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 8 variables explicatives
X_cols = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM", "PETROLE", "OR"]
X = data[X_cols]
#matrice de corrélation
corr = X.corr()
print(corr.round(2))

#carte de chaleur
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
plt.title("Corrélations entre les variables explicatives")
plt.tight_layout()
plt.show()

#VIF
X_const = sm.add_constant(X)
vif = pd.Series([variance_inflation_factor(X_const.values, i) for i in range(1, X_const.shape[1])],
                index=X_cols, name="VIF")
print(vif.round(2))

#Bloc8 entrainement et test
#variable y(à expliquer) et x(explicatives)
y = data["EXCES"]
X = data[X_cols]
#Coupure chronologique
X_train, y_train = X.loc[:"2023-12"], y.loc[:"2023-12"]
X_test, y_test = X.loc["2024-01":], y.loc["2024-01":]
#Vérifications
print(X_train.shape, X_test.shape)
print(X_train.index[0], X_train.index[-1])
print(X_test.index[0], X_test.index[-1])
print(pd.DataFrame({"entraînement": y_train.agg(["mean", "std"]), 
                    "test": y_test.agg(["mean", "std"])}).round(4))

#Bloc 9 MCO
#estimer modèle sur l'entraînement
modele = sm.OLS(y_train, sm.add_constant(X_train)).fit()
print(modele.summary())
#Application du modèle à l'échantillon test
prev = modele.predict(sm.add_constant(X_test)).values
rmse = np.sqrt(((y_test.values - prev) ** 2).mean())
r2_test = 1 - ((y_test.values - prev) ** 2).sum() / ((y_test.values - y_train.mean()) ** 2).sum()
print("RMSE test :", round (rmse, 4), "| R² test :", round(r2_test, 3))

import matplotlib.pyplot as plt
#instabilité des coefficients MCO
# fenêtre de 60 mois sur 136 mois
fenetre = 60
X_all = data[X_cols]
y_all = data["EXCES"]
#Régression MCO par fenêtre
liste_coef = []
dates_fin = []
for i in range(fenetre, len(data) + 1):
    Xi = X_all.iloc[i - fenetre:i]
    yi = y_all.iloc[i - fenetre:i]
    Ai = np.column_stack([np.ones(fenetre), Xi.values])
    bi = np.linalg.solve(Ai.T @ Ai, Ai.T @ yi.values)
    liste_coef.append(bi)
    dates_fin.append(Xi.index[-1])

coef_glis = pd.DataFrame(liste_coef, index=dates_fin, columns=["ALPHA"] + X_cols)
print(coef_glis.shape)
print(coef_glis.index[0], coef_glis.index[-1])

# Graphiques
coef_glis[["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]].plot(
    figsize=(10, 5), title="Coefficients glissants (60 mois) : facteurs")
plt.axhline(0, color="black", linewidth=0.8)
plt.show()

coef_glis[["PETROLE", "OR"]].plot(
    figsize=(10, 4), title="Coefficients glissants (60 mois) : pétrole et or")
plt.axhline(0, color="black", linewidth=0.8)
plt.show()

#Mesure de l'instabilité
print(coef_glis.agg(["min", "max", "std"]).T.round(3))

#Vérification de la signature de la colinéarité entre HML et CMA
print(coef_glis["HML"].corr(coef_glis["CMA"]).round(2))

# Ridge

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import RidgeCV

# Fonction d'évaluation réutilisable pour tous les modèles
def evaluer(y_vrai, y_prev, y_ref):
    rmse = np.sqrt(((y_vrai - y_prev) ** 2).mean())
    r2 = 1 - ((y_vrai - y_prev) ** 2).sum() / ((y_vrai - y_ref) ** 2).sum()
    return rmse, r2

resultats = {"MCO": evaluer(y_test.values, prev, y_train.mean())}

# Standardiser les variables avec l'entraînement seulement
scaler = StandardScaler().fit(X_train)
Z_train = scaler.transform(X_train)
Z_test = scaler.transform(X_test)

# Ridge : lambda choisi par validation croisée temporelle
lambdas = np.logspace(-2, 3, 100)
cv = TimeSeriesSplit(n_splits=5)
ridge = RidgeCV(alphas=lambdas, cv=cv, scoring="neg_root_mean_squared_error").fit(Z_train, y_train)
print("lambda retenu :", round(ridge.alpha_, 2))

# Revenir aux unités d'origine pour comparer avec les MCO
coef_ridge = ridge.coef_ / scaler.scale_
const_ridge = ridge.intercept_ - (coef_ridge * scaler.mean_).sum()

# Comparaison des coefficients
comparaison = pd.DataFrame(
    {"MCO": modele.params.values,
     "Ridge": np.concatenate([[const_ridge], coef_ridge])},
    index=["ALPHA"] + X_cols)
print(comparaison.round(3))

# Évaluation sur le test
prev_ridge = ridge.predict(Z_test)
resultats["Ridge"] = evaluer(y_test.values, prev_ridge, y_train.mean())
print(pd.DataFrame(resultats, index=["RMSE test", "R² test"]).T.round(4))


#Chemin de régularisation : coefficients (standardisés) pour chaque valeur de lambda

from sklearn.linear_model import Ridge, RidgeCV

chemin = pd.DataFrame(
    [Ridge(alpha=lam).fit(Z_train, y_train).coef_ for lam in lambdas],
    index=lambdas, columns=X_cols)

chemin.plot(logx=True, figsize=(10, 5), title="Chemin de régularisation de Ridge (variables standardisées)")
plt.axvline(ridge.alpha_, color="black", linestyle="--")
plt.axhline(0, color="gray", linewidth=0.8)
plt.xlabel("lambda (échelle logarithmique)")
plt.ylabel("coefficient")
plt.show()

# Ridge sur fenêtres glissantes, avec lambda ajusté à la taille de la fenêtre
lam_fen = ridge.alpha_ * fenetre / len(X_train)
liste_ridge = []
for i in range(fenetre, len(data) + 1):
    Xi = X_all.iloc[i - fenetre:i]
    yi = y_all.iloc[i - fenetre:i]
    ri = Ridge(alpha=lam_fen).fit(scaler.transform(Xi), yi)
    ci = ri.coef_ / scaler.scale_
    ki = ri.intercept_ - (ci * scaler.mean_).sum()
    liste_ridge.append(np.concatenate([[ki], ci]))

coef_glis_ridge = pd.DataFrame(liste_ridge, index=dates_fin, columns=["ALPHA"] + X_cols)

# Stabilité : écart-type des coefficients à travers les 77 fenêtres
stab = pd.DataFrame({"MCO": coef_glis.std(), "Ridge": coef_glis_ridge.std()})
stab["réduction (%)"] = ((1 - stab["Ridge"] / stab["MCO"]) * 100).round(0)
print(stab.round(3))

# Les coefficients de HML et CMA bougent-ils encore en sens opposé ?
print("corr HML-CMA  MCO :", coef_glis["HML"].corr(coef_glis["CMA"]).round(2),
      "| Ridge :", coef_glis_ridge["HML"].corr(coef_glis_ridge["CMA"]).round(2))