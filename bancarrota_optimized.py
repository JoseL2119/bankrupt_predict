# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:55:21 2026

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


# Ajustar el clasificador en el conjunto de entrenamiento usando Random Forest con SMOTE
# SMOTE es una técnica similar a las anteriores en cuanto a su objetivo. Esta lo que hace es identificar la clase 
# minoritaria y generar ejemplos sintéticos de la misma utilizando nearest neighbors. De esta manera, hace que la cantidad
# de observaciones de la clase minoritaria se acerque a la cantidad de la clase mayoritaria.
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
classifier_smote = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2)
classifier_smote.fit(X_train_resampled, y_train_resampled)


# Predicción de los Resultados con el conjunto de testing
y_pred_smote = classifier_smote.predict(X_test)

# Elaborar una matriz de confusión
cm_smote = confusion_matrix(y_test, y_pred_smote)





# Aplicar la mejora del RandomizedSearch para optimizar el modelo y sus parámetros
parameters = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(),
    param_distributions=parameters,
    n_iter=30,
    scoring='f1',  # IMPORTANTE
    cv=5,
    n_jobs=-1
)

random_search.fit(X_train_resampled, y_train_resampled)

best_score = random_search.best_score_
best_params = random_search.best_params_
print("Mejor score: ", random_search.best_score_)
print("\n\nMejores parámetros: ", random_search.best_params_)

best_model = random_search.best_estimator_

print("\n\nMejor modelo: ", best_model)

results_df = pd.DataFrame(random_search.cv_results_)


results_df = results_df.sort_values(by="mean_test_score", ascending=False)
top_results = results_df.head(10)

plt.figure(figsize=(10,6))
plt.barh(range(len(top_results)), top_results['mean_test_score'])
plt.yticks(range(len(top_results)), top_results['params'])
plt.xlabel("Score")
plt.title("Top 10 Modelos - Random Search")
plt.gca().invert_yaxis()
plt.show()

sns.boxplot(x=results_df['param_n_estimators'], y=results_df['mean_test_score'])
plt.title("Impacto de n_estimators")
plt.show()


classifier_smote_opt = RandomForestClassifier(n_estimators=200)
classifier_smote_opt.fit(X_train_resampled, y_train_resampled)


# Predicción de los Resultados con el conjunto de testing
y_pred_smote_opt = classifier_smote_opt.predict(X_test)

# Elaborar una matriz de confusión
cm_smote_opt = confusion_matrix(y_test, y_pred_smote_opt)



# Visualización mediante histogramas y otras tablas de los resultados
sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Confusion Matrix - Random Forest con SMOTE")
plt.show()

# Visualización mediante histogramas y otras tablas de los resultados
sns.heatmap(cm_smote_opt, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Confusion Matrix - Random Forest con SMOTE Optimizado")
plt.show()


from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score
print("\n\nPrecisión del modelo con SMOTE: ", accuracy_score(y_test, y_pred_smote))
print("\nRecall score con SMOTE: ", recall_score(y_test, y_pred_smote))
print("\nF1 score con SMOTE: ", f1_score(y_test, y_pred_smote))
print("\n\nReporte de estadísticas del modelo:\n", classification_report(y_test, y_pred_smote))



print("\n\nPrecisión del modelo con SMOTE: ", accuracy_score(y_test, y_pred_smote_opt))
print("\nRecall score con SMOTE: ", recall_score(y_test, y_pred_smote_opt))
print("\nF1 score con SMOTE: ", f1_score(y_test, y_pred_smote_opt))
print("\n\nReporte de estadísticas del modelo:\n", classification_report(y_test, y_pred_smote_opt))





import pickle


# Guardar el dataset de entrenamiento luego de ser escalado
pickle.dump(best_score, open("src/best_score.pkl", "wb"))

# Guardar modelo
pickle.dump(best_params, open("src/best_params.pkl", "wb"))

# Guardar matriz de confusión
pickle.dump(best_model, open("src/best_model.pkl", "wb"))

# Guardar características importantes
pickle.dump(results_df, open("src/results_df.pkl", "wb"))

# Guardar características importantes
pickle.dump(top_results, open("src/top_results.pkl", "wb"))

# Guardar características importantes
pickle.dump(cm_smote_opt, open("src/cm_smote_opt.pkl", "wb"))

# Guardar estadísticas del modelo
stats_opt = classification_report(y_test, y_pred_smote_opt, output_dict=True)
pickle.dump(stats_opt, open("src/stats_opt.pkl", "wb"))