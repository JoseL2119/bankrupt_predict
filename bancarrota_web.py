# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:20:51 2026

@author: Lenovo-PC
"""

# Representación de los resultados utilizando una librería para generar interfaces web
import streamlit as st
import numpy as np  # Para manejo de vectores y matrices
import matplotlib.pyplot as plt  # Para visualización de datos (aunque no se usa aquí)
import pandas as pd  # Para manipulación y análisis de datos
from matplotlib.colors import ListedColormap
import seaborn as sns
import plotly.express as px
import pickle

# Importar el dataset
dataset = pd.read_csv('data.csv')
dataset.columns = dataset.columns.str.strip()

st.title("Proyecto de predicción de Bancarrota 🏦")
st.write("Este proyecto consiste en predecir si una empresa termina en bancarrota, o no, a partir del siguiente conjunto de datos:")


st.subheader("Dataset a utilizar para el análisis")
st.write(dataset)
st.write("Contamos con información de aproximadamente 6000 empresas, con más de 90 métricas medidas.")

st.subheader("Análisis exploratorio")
st.write("Primero que nada, debemos ver la estructura que tiene el dataset.")

info_df = pd.DataFrame({
    "Column": dataset.columns,
    "Non-Null Count": dataset.notnull().sum().values,
    "Dtype": dataset.dtypes.values
})

st.dataframe(info_df)
st.write("Acá podemos observar los tipos de datos que tenemos y que no tenemos valores nulos en el dataset.")

st.write(dataset.describe())
st.write("Acá podemos ver información estadística de las variables como la media, la desviación estándar, los valores mínimos y máximos, etc.")
st.write("Lo que podemos interpretar de acá es que muchas variables tienen sus valores en rangos de 0 a 1, pero hay algunas otras variables que tienen valores que se disparan demasiado, cosa que puede ser un problema.")
st.write("A continuación veremos, mediante un histograma, la cantidad de empresas que tenemos en bancarrota.")
# Histograma de conteo de sobrevivientes
fig = px.histogram(dataset, x="Bankrupt?", color="Bankrupt?",
                   title="Conteo de Empresas")

st.plotly_chart(fig)

st.write("Podemos observar que la variable a predecir está completamente desbalanceada, con solo 220 empresas en bancarrota de 6819 observaciones totales. Esto es algo que también puede ser un problema.")
st.write("Ahora haremos un análisis de correlación para ver la relación lineal que hay entre las variables con respecto a la variable que queremos predecir para ver cuáles, a priori, son las que mayor influencia tienen y poderlas visualizar gráficamente.")
corr_matrix = dataset.corr()

fig, ax = plt.subplots(figsize=(20, 15))  # ajusta tamaño aquí

sns.heatmap(corr_matrix, cmap="coolwarm", ax=ax)

ax.set_title("Matriz de Correlación")

st.pyplot(fig)

st.write("De momento, vamos a fijarnos solo en la columna de la variable que queremos predecir, y nos vamos a quedar con las variables que tengan un valor de significancia absoluto mayor a 0.15.")
corr_features = corr_matrix["Bankrupt?"].abs() >= 0.15
feature_names = list(corr_features[corr_features].index)
st.write(feature_names)

st.write("Teniendo en cuenta solo estas variables, la matriz de significancia nos queda de la siguiente manera.")
mini_corr_matrix = dataset[feature_names].corr()

fig, ax = plt.subplots(figsize=(25, 15))  # ajusta tamaño aquí

sns.heatmap(mini_corr_matrix, annot=True, cmap="coolwarm", ax=ax)

ax.set_title("Matriz de Correlación")

st.pyplot(fig)

st.write("Lo que podemos interpretar de acá es que, teniendo en cuenta un valor de significancia de 0.15, solo 21 variables tienen una relación, por lo menos débil, con la variable a predecir. Sin embargo, quedarnos solo con estas variables para armar el modelo puede no ser el mejor de los enfoques, ya que con la correlación solo estamos evaluando de forma lineal las relaciones entre características.")
st.write("A partir de las variables que tenemos, vamos a hacer algunos gráficos para visualizar el comportamiento de nuestra variable predictora con respecto a las mismas.")

st.subheader("Distribución de Empresas en bancarrota según su Net Income")
fig, ax = plt.subplots(figsize=(15,15))
sns.histplot(data=dataset, x='Net Income to Total Assets', hue='Bankrupt?', bins=50, kde=True)
ax.set_title("Empresas en bancarrota según su Net Income")
st.pyplot(fig)
st.write("Mide qué tan rentable es la empresa usando lo que tiene. Podemos ver cómo las empresas en bancarrota suelen estar por debajo de la media, es decir, que son menos rentables.")



st.subheader("Distribución de Empresas en bancarrota según su Conversión de activos en ganancias después de impuestos")
fig, ax = plt.subplots(figsize=(15,15))
sns.histplot(data=dataset, x='ROA(A) before interest and % after tax', hue='Bankrupt?', bins=50, kde=True)
ax.set_title("Empresas en bancarrota según su conversión de activos en ganancias después de impuestos")
st.pyplot(fig)
st.write("Mide qué tan bien la empresa convierte activos en ganancias después de impuestos. Podemos ver cómo las empresas en bancarrota suelen estar por debajo de la media, es decir, son peores convirtiendo activos en ganancias.")


st.subheader("Distribución de Empresas en bancarrota según su Conversión de activos en ganancias incluyendo depreciación")
fig, ax = plt.subplots(figsize=(15,15))
sns.histplot(data=dataset, x='ROA(B) before interest and depreciation after tax', hue='Bankrupt?', bins=50, kde=True)
ax.set_title("Empresas en bancarrota según su conversión de activos en ganancias incluyendo depreciación")
st.pyplot(fig)
st.write("Mide qué tan bien la empresa convierte activos en ganancias incluyendo depreciación. Podemos ver cómo las empresas en bancarrota suelen estar por debajo de la media, es decir, son peores convirtiendo activos en ganancias.")


st.subheader("Distribución de Empresas en bancarrota según su Conversión de activos en ganancias antes de intereses")
fig, ax = plt.subplots(figsize=(15,15))
sns.histplot(data=dataset, x='ROA(C) before interest and depreciation before interest', hue='Bankrupt?', bins=50, kde=True)
ax.set_title("Empresas en bancarrota según su conversión de activos en ganancias antes de intereses")
st.pyplot(fig)
st.write("Mide qué tan bien la empresa convierte activos en ganancias antes de intereses. Podemos ver cómo las empresas en bancarrota suelen estar por debajo de la media, es decir, son peores convirtiendo activos en ganancias.")


st.subheader("Distribución de Empresas en bancarrota según su Patrimonio/Activos")
fig, ax = plt.subplots(figsize=(15,15))
sns.histplot(data=dataset, x='Net worth/Assets', hue='Bankrupt?', bins=50, kde=True)
ax.set_title("Empresas en bancarrota según su Patrimonio/Activos")
st.pyplot(fig)
st.write("Mide qué tanto de la empresa es realmente propia. Podemos ver cómo las empresas en bancarrota suelen estar por debajo de la media, es decir, parte de sus activos/patromio no es realmente de los dueños de la empresa.")


st.subheader("Distribución de Empresas en bancarrota según su porcentaje de Deuda sobre activos")
fig, ax = plt.subplots(figsize=(15,15))
sns.histplot(data=dataset, x='Debt ratio %', hue='Bankrupt?', bins=50, kde=True)
ax.set_title("Empresas en bancarrota según su porcentaje de deuda sobre activos")
st.pyplot(fig)
st.write("Mide qué tanto porcentaje de deuda tiene la empresa sobre sus activos. Podemos ver cómo las empresas en bancarrota suelen estar por encima de la media, es decir, tienen un porcentaje de deuda mayor, cosa que impacta negativamente en la esperanza de vida de una empresa.")


st.subheader("Distribución de Empresas en bancarrota según sus ganancias consistentes en el tiempo")
fig, ax = plt.subplots(figsize=(15,15))
sns.histplot(data=dataset, x='Persistent EPS in the Last Four Seasons', hue='Bankrupt?', bins=50, kde=True)
ax.set_title("Empresas en bancarrota según su porcentaje de deuda sobre activos")
st.pyplot(fig)
st.write("Mide qué tan bueno ha sido el rendimiento de la empresa en los últimos 4 años. Podemos ver cómo las empresas en bancarrota suelen estar por debajo de la media, es decir, han tenido un peor rendimiento en los últimos 4 años.")



st.subheader("Distribución de Empresas en bancarrota según sus ganancias acumuladas")
fig, ax = plt.subplots(figsize=(15,15))
sns.histplot(data=dataset, x='Retained Earnings to Total Assets', hue='Bankrupt?', bins=50, kde=True)
ax.set_title("Empresas en bancarrota según sus ganancias acumuladas")
st.pyplot(fig)
st.write("Mide qué cantidad de ganancias acumuladas tiene la empresa. Podemos ver cómo las empresas en bancarrota suelen estar por debajo de la media, es decir, tienen menor cantidad de ganancias acumuladas.")



st.subheader("Distribución de Empresas en bancarrota según sus beneficios a partir del capital invertido")
fig, ax = plt.subplots(figsize=(15,15))
sns.histplot(data=dataset, x='Net profit before tax/Paid-in capital', hue='Bankrupt?', bins=50, kde=True)
ax.set_title("Empresas en bancarrota según sus beneficios a partir del capital invertido")
st.pyplot(fig)
st.write("Mide la eficiencia del capital invertido en la empresa. ¿Qué tanta ganancia ha generado a partir de X cantidad de capital invertido? Podemos ver cómo las empresas en bancarrota suelen estar por debajo de la media, es decir, tienen menores ganancias a partir del capital invertido.")



st.subheader("Distribución de Empresas en bancarrota según su ganancia por acción")
fig, ax = plt.subplots(figsize=(15,15))
sns.histplot(data=dataset, x='Per Share Net profit before tax (Yuan ¥)', hue='Bankrupt?', bins=50, kde=True)
ax.set_title("Empresas en bancarrota según su ganancia por acción")
st.pyplot(fig)
st.write("Mide qué tanto se gana por tener acciones en la empresa. Podemos ver cómo las empresas en bancarrota suelen estar por debajo de la media, es decir, sus acciones pagan menos, lo que indica poca rentabilidad.")



st.subheader("Distribución de Empresas en bancarrota según su deuda a corto plazo en función de sus activos")
fig, ax = plt.subplots(figsize=(15,15))
sns.histplot(data=dataset, x='Current Liability to Assets', hue='Bankrupt?', bins=50, kde=True)
ax.set_title("Empresas en bancarrota según su deuda a corto plazo en función de sus activos")
st.pyplot(fig)
st.write("Mide qué tanta deuda tiene una empresa a corto plazo en función de su cantidad de activos. Podemos ver cómo las empresas en bancarrota suelen estar por encima de la media, es decir, suelen tener mayor cantidad de deuda en función de su cantidad de activos.")



st.subheader("Distribución de Empresas en bancarrota según su dependencia de financiamiento externo")
fig, ax = plt.subplots(figsize=(15,15))
sns.histplot(data=dataset, x='Borrowing dependency', hue='Bankrupt?', bins=50, kde=True)
ax.set_title("Empresas en bancarrota según su dependencia al financiamiento externo")
st.pyplot(fig)
st.write("Mide qué tan dependiente es una empresa al financiamiento externo. Podemos ver cómo las empresas en bancarrota suelen estar por encima de la media, es decir, son más dependientes del financiamiento externo, lo que se traduce en una deuda constante.")

st.write("A través de estos gráficos podemos ver cómo es el comportamiento de nuestra variable predictora en función de ciertas variables que indican el rendimiento de la empresa. Podemos ver cómo las empresas que acumulan mayor cantidad de deuda suelen ser más propensas a estar en bancarrota, mientras que las empresas que tienen un mejor rendimiento con el paso del tiempo suelen ser más saludables, cosa que es evidente.")
st.write("Otros detalles que pudimos identificar a través de este análisis es que nuestro dataset tiene valores atípicos y la variable a predecir está desproporcionada (muchas empresas sanas y pocas en bancarrota). De esta manera, el modelo a utilizar será el de Random Forest, ya que:")
st.write("- Maneja bien el ruido en los datos.")
st.write("- Maneja bien valores atípicos y variables desequilibradas. El sistema de votación a partir de la gran cantidad de árboles ayuda a reducir el sesgo hacia la clase mayoritaria, al mismo tiempo que hará que el modelo sea menos sensible a valores atípicos.")
st.write("- Maneja bien datasets con muchas variables. El modelo se encargará de seleccionar aleatoriamente características y darles puntuaciones de importancia, de modo que se terminará quedando con las variables realmente relevantes y eliminará a las irrelevantes.")

st.write("Sin embargo, podemos apoyarnos en algunas técnicas para mejorar aún más la efectivad de nuestro modelo en cuanto al desbalance entre clases. La técnica que utilizaremos se llama Resampling. Esta técnica consiste en reducir la diferencia entre las clases desproporcionadas aplicando estrategias como aumento de datos en la clase inferior, o disminución de datos en la clase superior.")
st.write("A la hora de evaluar el rendimiento de nuestro modelo nos estaremos enfocando en las métricas de Recall y el F1-Score, esto con el objetivo de reducir la cantidad de falsos negativos (empresas que se predicen como que no quebrarán, cuando en realidad si lo harán). El F1-Score se busca aumentar también para que asegurar un equilibrio entre la precisión y el recall.")
st.write("Teniendo esto en cuenta, podemos proceder a construir y entrenar nuestro modelo.")
st.divider()

# Explicando cómo se llevó a cabo el proceso de construcción del modelo
st.subheader("Construcción y entrenamiento del modelo")
st.write("El primer paso es el del pre-procesamiento de los datos, donde nos vamos a quedar solo con las variables que, a priori, realmente nos interesan, además de transformar las variables categóricas a variables numéricas que puedan ser interpretadas por nuestro modelo.")
st.write("Realizamos una división del dataset en 2. Por un lado, nos quedaremos con nuestras variables independientes, y por el otro, con la variable dependiente, que es la que queremos predecir, obteniendo el siguiente resultado.")
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

st.subheader("Dataset de variables independientes")
st.write(f"Shape de X: {X.shape}")
st.subheader("Dataset de la variable a predecir")
st.write(f"Shape de y: {y.shape}")

st.write("En el dataset de las variables independientes nos quedamos con todas las variables, por la razón comentada anteriormente.")
st.write("Podemos ver que no tenemos valores NAs ni variables categóricas, por lo que procederemos a hacer el escalado de variables.")

scaled_dataset = pickle.load(open("src/scaled_dataset.pkl", "rb"))
st.subheader("Dataset escalado (Conjunto de entrenamiento)")
st.write(scaled_dataset)
st.write("Ahora si, con todo el pre-procesamiento listo, podemos pasar a entrenar a nuestro modelo.")


# MOSTRANDO AHORA LOS RESULTADOS DEL MODELO
# Importamos el modelo y los resultados
cm = pickle.load(open("src/cm.pkl", "rb"))
stats = pickle.load(open("src/stats.pkl", "rb"))
stats_df = pd.DataFrame(stats).transpose()


st.subheader("Resultados del Modelo - Random Forest")

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)

ax.set_title("Matriz de Confusión")
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")

st.pyplot(fig)

st.write("Estadísticas del modelo")
st.dataframe(stats_df.style.format("{:.2f}"))

st.write("La interpretación que podemos darle a estos resultados es que se cumple un poco lo que mencionamos al principio. El modelo, en general, tiene una precisión que se podría considerar como buena, pero está sucediendo que tenemos muchos falsos negativos (predicciones de empresas como que no van a quebrar, cuando en realidad sí lo hicieron). A partir de este resultado, podemos decir que el Random Forest sin aplicar ninguna estrategia de Resampling no es una alternativa muy eficaz para resolver este problema.")

st.divider()


# MOSTRANDO LOS RESULTADOS DEL MODELO CON UNDERSAMPLING
# Importamos el modelo y los resultados
cm_rus = pickle.load(open("src/cm_rus.pkl", "rb"))
stats_rus = pickle.load(open("src/stats_rus.pkl", "rb"))
stats_df_rus = pd.DataFrame(stats_rus).transpose()


st.subheader("Resultados del Modelo - Random Forest con UnderSampling")

st.write("Antes de pasar con los resultados, me gustaría explicar brevemente como funciona esta técnica. El UnderSampling consiste en reducir la cantidad de puntos/datos pertenecientes a la clase mayoritaria, con el fin de que la cantidad de datos a evaluar sea similar para ambas clases. El objetivo de esta técnica siempre es reducir la brecha entre clases muy desiguales.")

fig, ax = plt.subplots()
sns.heatmap(cm_rus, annot=True, fmt='d', cmap='Blues', ax=ax)

ax.set_title("Matriz de Confusión")
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")

st.pyplot(fig)

st.write("Estadísticas del modelo")
st.dataframe(stats_df_rus.style.format("{:.2f}"))

st.write("Podemos observar que la predicción total del modelo disminuye un poco. La cantidad de falsos negativos disminuyó bastante, pero la cantidad de falsos positivos aumentó mucho y las predicciones correctas totales disminuyeron. Al usar este modelo, nuestro posible cliente se perdería la oportunidad de invertir en empresas que sí valen la pena por fallo de nuestro modelo, cosa que no nos podemos permitir. De esta manera, a pesar de que obtuvimos un mayor recall, la precisión general del modelo disminuyó demasiado, por lo que UnderSampling no es la estrategia correcta para solucionar este problema.")

st.divider()


# MOSTRANDO LOS RESULTADOS DEL MODELO CON OVERSAMPLING
# Importamos el modelo y los resultados
cm_ros = pickle.load(open("src/cm_ros.pkl", "rb"))
stats_ros = pickle.load(open("src/stats_ros.pkl", "rb"))
stats_df_ros = pd.DataFrame(stats_ros).transpose()


st.subheader("Resultados del Modelo - Random Forest con OverSampling")

st.write("Antes de pasar con los resultados, me gustaría explicar brevemente como funciona esta técnica. El OverSampling consiste en aumentar la cantidad de observaciones correspondientes a la clase minoritaria para equiparar la diferencia entre ambas clases.")

fig, ax = plt.subplots()
sns.heatmap(cm_ros, annot=True, fmt='d', cmap='Blues', ax=ax)

ax.set_title("Matriz de Confusión")
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")

st.pyplot(fig)

st.write("Estadísticas del modelo")
st.dataframe(stats_df_ros.style.format("{:.2f}"))

st.write("En este caso podemos observar que los resultados se mantienen similares a los obtenidos con Random Forest sin aplicar Resampling, incluso las métricas deseadas a evaluar empeoraron ligeramente. Habría que hacer más pruebas con diferentes muestras en ambos modelos para ver cuál de los dos se comporta mejor.")

st.divider()

# MOSTRANDO LOS RESULTADOS DEL MODELO CON SMOTE
# Importamos el modelo y los resultados
cm_smote = pickle.load(open("src/cm_smote.pkl", "rb"))
stats_smote = pickle.load(open("src/stats_smote.pkl", "rb"))
stats_df_smote = pd.DataFrame(stats_smote).transpose()


st.subheader("Resultados del Modelo - Random Forest con SMOTE")

st.write("Antes de pasar con los resultados, me gustaría explicar brevemente como funciona esta técnica. SMOTE consiste en aumentar la cantidad de datos correspondientes a la clase minoritaria utilizando ejemplos sintéticos a través de la técnica de Nearest Neighbors.")

fig, ax = plt.subplots()
sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Blues', ax=ax)

ax.set_title("Matriz de Confusión")
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")

st.pyplot(fig)

st.write("Estadísticas del modelo")
st.dataframe(stats_df_smote.style.format("{:.2f}"))

st.write("Con esta última técnica los resultados obtenidos son mejores. Aunque el porcentaje de precisión para la clase 1 disminuye un poco con respecto a Random Forest sin aplicar Resampling, las métricas que nos interesan (Recall y F1-Score) aumentaron notablemente, lo que quiere decir que reducimos el riesgo de invertir en empresas que van a quebrar, por lo que podemos concluir que SMOTE es el mejor enfoque para trabajar este problema debido a que es el que mejores resultados nos otorgó en cuanto a las métricas que deseamos evaluar desde un inicio.")


st.divider()

st.subheader("Variables más influyentes según el modelo de Random Forest")

importances = pickle.load(open("src/importances.pkl", "rb"))

X_2 = dataset.drop('Bankrupt?', axis=1)

top_n = 20

feat_importance = pd.Series(importances, index=X_2.columns)

top_features = feat_importance.sort_values(ascending=False).head(top_n)

fig, ax = plt.subplots()

top_features.sort_values().plot(
    kind='barh',
    figsize=(10, 6)
)

ax.set_title("Visualización de variables más influyentes - RF")

st.pyplot(fig)



st.subheader("Variables más influyentes según el modelo de Random Forest con UnderSampling")

importances_rus = pickle.load(open("src/importances_rus.pkl", "rb"))

top_n = 20

feat_importance_rus = pd.Series(importances_rus, index=X_2.columns)

top_features_rus = feat_importance_rus.sort_values(ascending=False).head(top_n)

fig, ax = plt.subplots()

top_features_rus.sort_values().plot(
    kind='barh',
    figsize=(10, 6)
)

ax.set_title("Visualización de variables más influyentes - RF UnderSampling")

st.pyplot(fig)






st.subheader("Variables más influyentes según el modelo de Random Forest con OverSampling")

importances_ros = pickle.load(open("src/importances_ros.pkl", "rb"))

top_n = 20

feat_importance_ros = pd.Series(importances_ros, index=X_2.columns)

top_features_ros = feat_importance_ros.sort_values(ascending=False).head(top_n)

fig, ax = plt.subplots()

top_features_ros.sort_values().plot(
    kind='barh',
    figsize=(10, 6)
)

ax.set_title("Visualización de variables más influyentes - RF OverSampling")

st.pyplot(fig)






st.subheader("Variables más influyentes según el modelo de Random Forest con SMOTE")

importances_smote = pickle.load(open("src/importances_smote.pkl", "rb"))

top_n = 20

feat_importance_smote = pd.Series(importances_smote, index=X_2.columns)

top_features_smote = feat_importance_smote.sort_values(ascending=False).head(top_n)

fig, ax = plt.subplots()

top_features_smote.sort_values().plot(
    kind='barh',
    figsize=(10, 6)
)

ax.set_title("Visualización de variables más influyentes - RF SMOTE")

st.pyplot(fig)


st.divider()

st.write("""A pesar del dataset con valores atípicos y descompensación entre clases predictoras, se hizo un 
         trabajo interesante en la selección base del modelo a utilizar, y luego con las estrategias de Resampling
         implementadas. Random forest demostró tener un rendimiento, a priori, superior por sí solo que utilizando
         técnicas como UnderSampling u OverSampling, pero SMOTE demotrostó ser la mejor alternativa en cuanto
         a las métricas que se buscaban mejorar desde un inicio, a pesar de no tener la mismo precisión que
         tenía Random Forest. De esta manera, podemos concluir que el modelo de Random Forest utilizando la técnica
         SMOTE es la mejor alternativa a seguir para trabajar este problema.""")
         
st.write("""Creo que el algoritmo de Random Forest con SMOTE es optimizable en cuanto a la selección de hiperparámetros
         utilizando técnicas como Grid Search o Randomized Search, de modo que se pueda intentar mejorar
         al máximo el rendimiento del modelo.
         """)


st.divider()

st.subheader("Intento de optimización del modelo utilizando RandomizedSearch")

parameters = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

st.write("En este caso, aplicamos un RandomizedSearch para buscar los mejores hiperparámetros del modelo, utilizando el siguiente diccionario de parámetros posibles.")
st.dataframe(parameters)

st.write("Los resultados obtenidos fueron los siguientes.")

best_score = pickle.load(open("src/best_score.pkl", "rb"))
best_params = pickle.load(open("src/best_params.pkl", "rb"))
best_model = pickle.load(open("src/best_model.pkl", "rb"))

st.write(f"Puntuación más alta alcanzada en función de la media de los resultados obtenidos: {best_score}")
st.write(f"Mejores Parámetros a utilizar en el modelo: {best_params}")
st.write(f"Mejor modelo resultante: {best_model}")
st.write("Se ve así porque el mejor modelo resultante fue el RF con los parámetros por defecto.")

import plotly.express as px
top_results = pickle.load(open("src/top_results.pkl", "rb"))

# Convertir params a texto más legible
labels = [
    f"n={p['n_estimators']}, features={p['max_features']}, depth={p['max_depth']}, split={p['min_samples_split']}"
    for p in top_results['params']
]

df_plot = pd.DataFrame({
    "Score": top_results['mean_test_score'],
    "Modelo": labels
})

fig = px.bar(
    df_plot,
    x="Score",
    y="Modelo",
    orientation='h',
    title="Top 10 Modelos - Random Search"
)

st.plotly_chart(fig)

st.write("Podemos ver que la diferencia entre los distintos modelos es prácticamente imperceptible. Esto quiere decir que el modelo que armamos previamente ya se encontraba cerca de lo óptimo. También podemos concluir que los hiperparámetros no tienen tanto impacto como el manejo del desbalace.")




