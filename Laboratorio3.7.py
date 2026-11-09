#!/usr/bin/env python
# coding: utf-8

# # Laboratorio sin calificación: Regresión logística con Scikit-Learn
# 
# 
# 

# ## Objetivos
# En este laboratorio vas a:
# - Entrenar un modelo de regresión logística con scikit-learn.
# 

# ## Conjunto de datos 
# Empecemos con el mismo conjunto de datos de antes.

# In[ ]:


import numpy as np

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])


# ## Ajustar el modelo
# 
# El código de abajo importa el [modelo de regresión logística](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression) de scikit-learn. Puedes ajustar este modelo a los datos de entrenamiento llamando a la función `fit`.

# In[ ]:


from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(X, y)


# ## Hacer predicciones
# 
# Puedes ver las predicciones de este modelo llamando a la función `predict`.

# In[ ]:


y_pred = lr_model.predict(X)

print("Predicción sobre el conjunto de entrenamiento:", y_pred)


# ## Calcular la exactitud
# 
# Puedes calcular la exactitud (accuracy) de este modelo llamando a la función `score`.

# In[ ]:


print("Exactitud sobre el conjunto de entrenamiento:", lr_model.score(X, y))
