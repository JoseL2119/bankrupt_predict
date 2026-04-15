# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 11:31:02 2026

@author: Lenovo-PC
"""

# PREDICCIÓN BANCARROTA

# Importando las bibliotecas necesarias
import numpy as np  # Para manejo de vectores y matrices
import matplotlib.pyplot as plt  # Para visualización de datos (aunque no se usa aquí)
import pandas as pd  # Para manipulación y análisis de datos
from matplotlib.colors import ListedColormap
import seaborn as sns

from imblearn.under_sampling import RandomUnderSampler # Resampling
from imblearn.over_sampling import RandomOverSampler 
from imblearn.over_sampling import SMOTE

# Importar el dataset
dataset = pd.read_csv('data.csv')

# El .info muy util para ver la estructura del dataset, los tipos de datos y si tiene datos NAs (y su cantidad).
dataset.info() 

# El .describe también muy útil para visualizar información estadísitca de las columnas del dataset.
dataset.describe()

dataset.columns = dataset.columns.str.strip()

"""
# ANÁLISIS EXPLORATORIO UTILIZANDO LA LIBRERÍA SEABORN
# Conteo de Empresas en Bancarrota
sns.countplot(x='Bankrupt?', data=dataset, palette='coolwarm')
plt.title("Conteo de Empresas en Bancarrota")
plt.show()

# HACEMOS UNA PRUEBA DE CORRELACIÓN PARA VER LAS VARIABLES QUE MÁS INFLUENCIA TIENEN SOBRE NUESTRA PREDICCIÓN 
corr = dataset.corr()
corr_target = corr['Bankrupt?'].drop('Bankrupt?')
corr_target_abs = corr_target.abs().sort_values(ascending=False)
top_vars = corr_target_abs.head(15).index # Nos quedamos con las 15 que más influyen

# Teniendo en cuenta dichas variables, podemos proceder con un mejor análisis exploratorio
# Distribución de empresas en bancarrota según su Net Income
# Mide qué tan rentable es la empresa usando lo que tiene
sns.histplot(data=dataset, x='Net Income to Total Assets', hue='Bankrupt?', bins=50, kde=True)
plt.title("Empresas en bancarrota según su Net Income")
plt.show()

# Distribución de empresas en bancarrota según su Return on Assets
# Mide qué tan bien la empresa convierte activos en ganancias
# Después de impuestos
sns.histplot(data=dataset, x='ROA(A) before interest and % after tax', hue='Bankrupt?', bins=50, kde=True)
plt.title("Survival by Passenger Class")
plt.show()

# Incluyendo depreciación
sns.histplot(data=dataset, x='ROA(B) before interest and depreciation after tax', hue='Bankrupt?', bins=50, kde=True)
plt.title("Survival by Passenger Class")
plt.show()

# Antes de intereses
sns.histplot(data=dataset, x='ROA(C) before interest and depreciation before interest', hue='Bankrupt?', bins=50, kde=True)
plt.title("Survival by Passenger Class")
plt.show()

# Patrimonio / activos - Qué parte es realmente “propia”
sns.histplot(data=dataset, x='Net worth/Assets', hue='Bankrupt?', bins=50, kde=True)
plt.title("Survival by Passenger Class")
plt.show()

# % de deuda sobre activos
sns.histplot(data=dataset, x='Debt ratio %', hue='Bankrupt?', bins=50, kde=True)
plt.title("Survival by Passenger Class")
plt.show()

# Ganancias consistentes en el tiempo
sns.histplot(data=dataset, x='Persistent EPS in the Last Four Seasons', hue='Bankrupt?', bins=50, kde=True)
plt.title("Survival by Passenger Class")
plt.show()

# Ganancias acumuladas
sns.histplot(data=dataset, x='Retained Earnings to Total Assets', hue='Bankrupt?', bins=50, kde=True)
plt.title("Survival by Passenger Class")
plt.show()

# Beneficio vs capital invertido - Mide eficiencia del capital
sns.histplot(data=dataset, x='Net profit before tax/Paid-in capital', hue='Bankrupt?', bins=50, kde=True)
plt.title("Survival by Passenger Class")
plt.show()

# Ganancia por acción
sns.histplot(data=dataset, x='Per Share Net profit before tax (Yuan ¥)', hue='Bankrupt?', bins=50, kde=True)
plt.title("Survival by Passenger Class")
plt.show()

# Deuda a corto plazo vs activos
sns.histplot(data=dataset, x='Current Liability to Assets', hue='Bankrupt?', bins=50, kde=True)
plt.title("Survival by Passenger Class")
plt.show()

# Capital de trabajo
sns.histplot(data=dataset, x='Working Capital to Total Assets', hue='Bankrupt?', bins=50, kde=True)
plt.title("Survival by Passenger Class")
plt.show()

# Return on Equity - Qué tanto ganan los accionistas?
sns.histplot(data=dataset, x="Net Income to Stockholder's Equity", hue='Bankrupt?', bins=50, kde=True)
plt.title("Survival by Passenger Class")
plt.show()

# Dependencia de financiamiento externo
sns.histplot(data=dataset, x='Borrowing dependency', hue='Bankrupt?', bins=50, kde=True)
plt.title("Survival by Passenger Class")
plt.show()

# Deudas vs activos líquidos
sns.histplot(data=dataset, x='Current Liability to Current Assets', hue='Bankrupt?', bins=50, kde=True)
plt.title("Survival by Passenger Class")
plt.show()
"""


# FASE DE PRE-PROCESADO    

X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
from sklearn.model_selection import train_test_split

# Dividimos los datos: 80% para entrenamiento y 20% para prueba
# random_state asegura que los resultados sean reproducibles
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Escalado de características
# Estandarizamos los valores para que tengan media 0 y desviación estándar 1
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # Ajustamos y transformamos las características numéricas del conjunto de entrenamiento
X_test = sc.transform(X_test) # Transformamos en el conjunto de test aplicando el mismo ajuste



# Como estamos frente a un dataset con valores atípicos y desbalanceados. La proporción entre empresas 
# que fueron a la quiebra y empresas que no, es muy grande. Debido a esto, se utiliza el algoritmo de Random Forest,
# que usa el método de votación y varios árboles, cosa que reduce el sesgo que se pueda tener hacia la clase mayoritaria.
# También será necesario implementar técnicas como el Resampling para combatir esta problemática y evitar falsos negativos,
# o lo que es lo mismo, que nuestro algoritmo marque empresas como que no van a quebrar, cuando en realidad sí van por ese camino.

# Ajustar el clasificador en el conjunto de entrenamiento usando Random Forest normal
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)


# Predicción de los Resultados con el conjunto de testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



# Ajustar el clasificador en el conjunto de entrenamiento usando Random Forest con UnderSampling
# El UnderSampling lo que hace es reducir el número de datos en la clase mayoritaria, haciendo que se acerque a la
# minoritaria. El objetivo que tiene es atacar el desbalance entre clases.
rus = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
classifier_rus = RandomForestClassifier(random_state=42)
classifier_rus.fit(X_train_resampled, y_train_resampled)


# Predicción de los Resultados con el conjunto de testing
y_pred_rus = classifier_rus.predict(X_test)

# Elaborar una matriz de confusión
cm_rus = confusion_matrix(y_test, y_pred_rus)






# Ajustar el clasificador en el conjunto de entrenamiento usando Random Forest con OverSampling
# El OverSampling, al contrario del método anterior, aumenta la cantidad de datos en la clase minoritaria, 
# acercándose así a la clase mayoritaria.
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
classifier_ros = RandomForestClassifier(random_state=42)
classifier_ros.fit(X_train_resampled, y_train_resampled)


# Predicción de los Resultados con el conjunto de testing
y_pred_ros = classifier_ros.predict(X_test)

# Elaborar una matriz de confusión
cm_ros = confusion_matrix(y_test, y_pred_ros)





# Ajustar el clasificador en el conjunto de entrenamiento usando Random Forest con SMOTE
# SMOTE es una técnica similar a las anteriores en cuanto a su objetivo. Esta lo que hace es identificar la clase 
# minoritaria y generar ejemplos sintéticos de la misma utilizando nearest neighbors. De esta manera, hace que la cantidad
# de observaciones de la clase minoritaria se acerque a la cantidad de la clase mayoritaria.
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
classifier_smote = RandomForestClassifier(random_state=42)
classifier_smote.fit(X_train_resampled, y_train_resampled)


# Predicción de los Resultados con el conjunto de testing
y_pred_smote = classifier_smote.predict(X_test)

# Elaborar una matriz de confusión
cm_smote = confusion_matrix(y_test, y_pred_smote)





# Visualización mediante histogramas y otras tablas de los resultados
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Confusion Matrix - Random Forest")
plt.show()

# Visualización mediante histogramas y otras tablas de los resultados
sns.heatmap(cm_rus, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Confusion Matrix - Random Forest con UnderSampling")
plt.show()

# Visualización mediante histogramas y otras tablas de los resultados
sns.heatmap(cm_ros, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Confusion Matrix - Random Forest con OverSampling")
plt.show()

# Visualización mediante histogramas y otras tablas de los resultados
sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Confusion Matrix - Random Forest con SMOTE")
plt.show()

from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score
print("Precisión del modelo sin hacer Resampling: ", accuracy_score(y_test, y_pred))
print("\nRecall score sin hacer Resampling: ", recall_score(y_test, y_pred))
print("\nF1 score sin hacer Resampling: ", f1_score(y_test, y_pred))
print("\n\nReporte de estadísticas del modelo:\n", classification_report(y_test, y_pred))

print("\n\nPrecisión del modelo con UnderSampling: ", accuracy_score(y_test, y_pred_rus))
print("\nRecall score con UnderSampling: ", recall_score(y_test, y_pred_rus))
print("\nF1 score con UnderSampling: ", f1_score(y_test, y_pred_rus))
print("\n\nReporte de estadísticas del modelo:\n", classification_report(y_test, y_pred_rus))

print("\n\nPrecisión del modelo con OverSampling: ", accuracy_score(y_test, y_pred_ros))
print("\nRecall score con OverSampling: ", recall_score(y_test, y_pred_ros))
print("\nF1 score con OverSampling: ", f1_score(y_test, y_pred_ros))
print("\n\nReporte de estadísticas del modelo:\n", classification_report(y_test, y_pred_ros))

print("\n\nPrecisión del modelo con SMOTE: ", accuracy_score(y_test, y_pred_smote))
print("\nRecall score con SMOTE: ", recall_score(y_test, y_pred_smote))
print("\nF1 score con SMOTE: ", f1_score(y_test, y_pred_smote))
print("\n\nReporte de estadísticas del modelo:\n", classification_report(y_test, y_pred_smote))



importances = classifier.feature_importances_

X_2 = dataset.drop('Bankrupt?', axis=1)

top_n = 20

feat_importance = pd.Series(classifier.feature_importances_, index=X_2.columns)

top_features = feat_importance.sort_values(ascending=False).head(top_n)

top_features.sort_values().plot(
    kind='barh',
    figsize=(10, 6)
)

plt.title(f"Top {top_n} Feature Importance - Random Forest")
plt.show()






importances_rus = classifier_rus.feature_importances_

top_n = 20

feat_importance_rus = pd.Series(classifier_rus.feature_importances_, index=X_2.columns)

top_features_rus = feat_importance_rus.sort_values(ascending=False).head(top_n)

top_features_rus.sort_values().plot(
    kind='barh',
    figsize=(10, 6)
)

plt.title(f"Top {top_n} Feature Importance - Random Forest UnderSampling")
plt.show()






importances_ros = classifier_ros.feature_importances_

top_n = 20

feat_importance_ros = pd.Series(classifier_ros.feature_importances_, index=X_2.columns)

top_features_ros = feat_importance_ros.sort_values(ascending=False).head(top_n)

top_features_ros.sort_values().plot(
    kind='barh',
    figsize=(10, 6)
)

plt.title(f"Top {top_n} Feature Importance - Random Forest OverSampling")
plt.show()







importances_smote = classifier_smote.feature_importances_

top_n = 20

feat_importance_smote = pd.Series(classifier_smote.feature_importances_, index=X_2.columns)

top_features_smote = feat_importance_smote.sort_values(ascending=False).head(top_n)

top_features_smote.sort_values().plot(
    kind='barh',
    figsize=(10, 6)
)

plt.title(f"Top {top_n} Feature Importance - Random Forest SMOTE")
plt.show()



# Creamos una tabla para evaluar el rendimiento
accuracy = [accuracy_score(y_test, y_pred), accuracy_score(y_test, y_pred_rus), accuracy_score(y_test, y_pred_ros), accuracy_score(y_test, y_pred_smote)]
recall = [recall_score(y_test, y_pred), recall_score(y_test, y_pred_rus), recall_score(y_test, y_pred_ros), recall_score(y_test, y_pred_smote)]
f1 = [f1_score(y_test, y_pred), f1_score(y_test, y_pred_rus), f1_score(y_test, y_pred_ros), f1_score(y_test, y_pred_smote)]
model = ['Random Forest Base', 'Random Forest con UnderSampling', 'Random Forest con OverSampling',
       'Random Forest con SMOTE',]

compare = pd.DataFrame({'Modelo': model, 'Precisión': accuracy, 'Recall': recall, 'F1-score': f1})
compare



import pickle


# Guardar el dataset de entrenamiento luego de ser escalado
pickle.dump(X_train, open("src/scaled_dataset.pkl", "wb"))

# Guardar modelo
pickle.dump(classifier, open("src/model.pkl", "wb"))

# Guardar matriz de confusión
pickle.dump(cm, open("src/cm.pkl", "wb"))

# Guardar estadísticas del modelo
stats = classification_report(y_test, y_pred, output_dict=True)
pickle.dump(stats, open("src/stats.pkl", "wb"))

# Guardar características importantes
pickle.dump(importances, open("src/importances.pkl", "wb"))




# Guardar modelo con UnderSampling
pickle.dump(classifier_rus, open("src/model_rus.pkl", "wb"))

# Guardar matriz de confusión
pickle.dump(cm_rus, open("src/cm_rus.pkl", "wb"))

# Guardar estadísticas del modelo
stats_rus = classification_report(y_test, y_pred_rus, output_dict=True)
pickle.dump(stats_rus, open("src/stats_rus.pkl", "wb"))

# Guardar características importantes
pickle.dump(importances_rus, open("src/importances_rus.pkl", "wb"))



# Guardar modelo con OverSampling
pickle.dump(classifier_ros, open("src/model_ros.pkl", "wb"))

# Guardar matriz de confusión
pickle.dump(cm_ros, open("src/cm_ros.pkl", "wb"))

# Guardar estadísticas del modelo
stats_ros = classification_report(y_test, y_pred_ros, output_dict=True)
pickle.dump(stats_ros, open("src/stats_ros.pkl", "wb"))

# Guardar características importantes
pickle.dump(importances_ros, open("src/importances_ros.pkl", "wb"))



# Guardar modelo con SMOTE
pickle.dump(classifier_smote, open("src/model_smote.pkl", "wb"))

# Guardar matriz de confusión
pickle.dump(cm_smote, open("src/cm_smote.pkl", "wb"))

# Guardar estadísticas del modelo
stats_smote = classification_report(y_test, y_pred_smote, output_dict=True)
pickle.dump(stats_smote, open("src/stats_smote.pkl", "wb"))

# Guardar características importantes
pickle.dump(importances_smote, open("src/importances_smote.pkl", "wb"))
