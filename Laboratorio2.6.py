#!/usr/bin/env python
# coding: utf-8

# # Laboratorio opcional: Regresión lineal con Scikit-Learn

# Existe una herramienta de machine learning de código abierto y de uso comercial llamada [scikit-learn](https://scikit-learn.org/stable/index.html). Esta herramienta incluye implementaciones de muchos de los algoritmos con los que trabajarás en este curso.
# 
# 

# ## Objetivos
# En este laboratorio vas a:
# - Usar scikit-learn para implementar regresión lineal con descenso de gradiente

# ## Herramientas
# Usarás funciones de scikit-learn, además de Matplotlib y NumPy. 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from lab_utils_multi import  load_house_data
from lab_utils_common import dlc
np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')


# # Descenso de gradiente
# Scikit-learn tiene un modelo de regresión con descenso de gradiente: [sklearn.linear_model.SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#examples-using-sklearn-linear-model-sgdregressor). Igual que tu implementación anterior del descenso de gradiente, este modelo funciona mejor con datos de entrada normalizados. [sklearn.preprocessing.StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler) aplica la normalización z-score, como en un laboratorio anterior. Aquí se le llama "puntuación estándar" (standard score).

# ### Cargar el conjunto de datos

# In[ ]:


X_train, y_train = load_house_data()
X_features = ['tamaño (pies²)','habitaciones','pisos','antigüedad']


# ### Escalar/normalizar los datos de entrenamiento

# In[ ]:


scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
print(f"Rango (máx - mín) por columna en X original:    {np.ptp(X_train,axis=0)}")   
print(f"Rango (máx - mín) por columna en X normalizado: {np.ptp(X_norm,axis=0)}")


# ### Crear y ajustar el modelo de regresión

# In[ ]:


sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
print(sgdr)
print(f"número de iteraciones completadas: {sgdr.n_iter_}, número de actualizaciones de pesos: {sgdr.t_}")


# ### Ver los parámetros
# Ten en cuenta que los parámetros corresponden a los datos de entrada *normalizados*. Los parámetros ajustados son muy parecidos a los que se obtuvieron con estos datos en el laboratorio anterior.

# In[ ]:


b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"parámetros del modelo:                       w: {w_norm}, b:{b_norm}")
print( "parámetros del modelo del laboratorio anterior: w: [110.56 -21.27 -32.71 -37.97], b: 363.16")


# ### Hacer predicciones
# Predice los valores objetivo de los datos de entrenamiento. Usa tanto la función `predict` como el cálculo directo con $w$ y $b$.

# In[ ]:


# hacer una predicción con sgdr.predict()
y_pred_sgd = sgdr.predict(X_norm)
# hacer una predicción con w,b 
y_pred = np.dot(X_norm, w_norm) + b_norm  
print(f"¿Coinciden las predicciones de np.dot() y sgdr.predict?: {(y_pred == y_pred_sgd).all()}")

print(f"Predicción sobre el conjunto de entrenamiento:\n{y_pred[:4]}" )
print(f"Valores objetivo \n{y_train[:4]}")


# ### Graficar los resultados
# Grafiquemos las predicciones contra los valores objetivo.

# In[ ]:


# graficar predicciones y objetivos contra las variables originales    
fig,ax=plt.subplots(1,4,figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'objetivo')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],y_pred,color=dlc["dlorange"], label = 'predicción')
ax[0].set_ylabel("Precio"); ax[0].legend();
fig.suptitle("objetivo vs. predicción con el modelo normalizado con z-score")
plt.show()


# ## ¡Felicidades!
# En este laboratorio:
# - Usaste scikit-learn, una herramienta de machine learning de código abierto.
# - Implementaste regresión lineal con descenso de gradiente y normalización de variables usando esa herramienta.

# In[ ]:
