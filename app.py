import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# configuração da página
st.set_page_config(page_title="HidroCore - Coagulante", layout="wide")
st.title("💧 Otimização da Dosagem de Coagulante com IA")

# gerar dados simulados
np.random.seed(42)
n = 500

dados = pd.DataFrame({
    "turbidez": np.random.uniform(5,100,n),
    "cor": np.random.uniform(10,200,n),
    "ph": np.random.uniform(6,8,n),
    "temperatura": np.random.uniform(15,30,n),
    "vazao": np.random.uniform(100,500,n)
})

dados["dosagem_otima"] = (
    0.4*dados["turbidez"] +
    0.2*dados["cor"] +
    5*(7 - dados["ph"]) +
    0.01*dados["vazao"] +
    np.random.normal(0,3,n)
)

# separar variáveis
X = dados.drop("dosagem_otima", axis=1)
y = dados["dosagem_otima"]

# dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# treinar modelo
modelo = RandomForestRegressor(n_estimators=400)
modelo.fit(X_train, y_train)

# -------------------------
# Avaliação do modelo
# -------------------------

y_pred = modelo.predict(X_test)

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)

# linha ideal
ax.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

ax.set_xlabel("Dosagem Observada")
ax.set_ylabel("Dosagem Prevista")
ax.set_title("Observado vs Previsto")

st.subheader("Desempenho do Modelo")
st.pyplot(fig)

r2 = r2_score(y_test, y_pred)
st.write("R² do modelo:", round(r2,3))

# -------------------------
# Interface de entrada
# -------------------------

st.sidebar.header("📡 Dados da Água Bruta")

turbidez = st.sidebar.slider("Turbidez",0,200,40)
cor = st.sidebar.slider("Cor",0,300,120)
ph = st.sidebar.slider("pH",6.0,8.5,7.2)
temperatura = st.sidebar.slider("Temperatura",10,35,24)
vazao = st.sidebar.slider("Vazão",50,600,250)

entrada = pd.DataFrame({
    "turbidez":[turbidez],
    "cor":[cor],
    "ph":[ph],
    "temperatura":[temperatura],
    "vazao":[vazao]
})

dosagem = modelo.predict(entrada)[0]

st.metric("Dosagem Ótima", f"{round(dosagem,2)} mg/L")

# -------------------------
# Importância das variáveis
# -------------------------

importances = modelo.feature_importances_

fig2, ax2 = plt.subplots()
ax2.bar(X.columns, importances)

st.subheader("Importância das Variáveis no Modelo")
st.pyplot(fig2)


