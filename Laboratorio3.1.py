#!/usr/bin/env python
# coding: utf-8

# # Laboratorio opcional: Clasificación
# 
# En este laboratorio vas a contrastar la regresión con la clasificación.

# In[1]:


import numpy as np
#get_ipython().run_line_magic('matplotlib', 'widget')
import matplotlib.pyplot as plt
from lab_utils_common import dlc, plot_data
from plt_one_addpt_onclick import plt_one_addpt_onclick
#plt.style.use('./deeplearning.mplstyle')


# ## Problemas de clasificación
# <img align="left" src="./images/C1_W3_Classification.png"     style=" width:380px; padding: 10px; " > Algunos ejemplos de problemas de clasificación son: identificar si un correo es spam o no, o determinar si un tumor es maligno o benigno. En particular, estos son ejemplos de clasificación *binaria*, donde hay dos resultados posibles. Los resultados se pueden describir en pares "positivo"/"negativo", como "sí"/"no", "verdadero"/"falso" o "1"/"0". 
# 
# Las gráficas de conjuntos de datos de clasificación suelen usar símbolos para indicar el resultado de cada ejemplo. En las gráficas de abajo, la "X" representa los resultados positivos y la "O", los negativos. 

# In[2]:


x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])
X_train2 = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train2 = np.array([0, 0, 0, 1, 1, 1])


# In[3]:


pos = y_train == 1
neg = y_train == 0

fig,ax = plt.subplots(1,2,figsize=(8,3))
#gráfica 1, una variable
ax[0].scatter(x_train[pos], y_train[pos], marker='x', s=80, c = 'red', label="y=1")
ax[0].scatter(x_train[neg], y_train[neg], marker='o', s=100, label="y=0", facecolors='none', 
              edgecolors=dlc["dlblue"],lw=3)

ax[0].set_ylim(-0.08,1.1)
ax[0].set_ylabel('y', fontsize=12)
ax[0].set_xlabel('x', fontsize=12)
ax[0].set_title('gráfica con una variable')
ax[0].legend()

#gráfica 2, dos variables
plot_data(X_train2, y_train2, ax[1])
ax[1].axis([0, 4, 0, 4])
ax[1].set_ylabel('$x_1$', fontsize=12)
ax[1].set_xlabel('$x_0$', fontsize=12)
ax[1].set_title('gráfica con dos variables')
ax[1].legend()
plt.tight_layout()
plt.show()


# Observa en las gráficas:
# - En la gráfica de una variable, los resultados positivos se muestran como "X" rojas y en y=1. Los negativos son "O" azules y están en y=0.
#    - Recuerda que, en regresión lineal, y no estaba limitada a dos valores, sino que podía tomar cualquier valor.
# - En la gráfica de dos variables no hay eje para y. Los resultados positivos se muestran como "X" rojas y los negativos con el símbolo "O" azul.
#     - Recuerda que, en regresión lineal con múltiples variables, y no estaba limitada a dos valores, y una gráfica similar habría sido tridimensional.

# ## Enfoque con regresión lineal
# La semana pasada aplicaste regresión lineal para construir un modelo de predicción. Probemos ese enfoque aquí con el ejemplo sencillo que se describió en clase. El modelo predecirá si un tumor es benigno o maligno según su tamaño. Prueba lo siguiente:
# - Haz clic en "Run Linear Regression" (ejecutar regresión lineal) para encontrar el mejor modelo de regresión lineal para los datos.
#     - Observa que el modelo lineal resultante **no** se ajusta bien a los datos. 
# Una forma de mejorar los resultados es aplicar un *umbral*. 
# - Marca la casilla "Toggle 0.5 threshold" (activar umbral de 0.5) para ver las predicciones al aplicar el umbral.
#     - Estas predicciones se ven bien: coinciden con los datos.
# - *Importante*: ahora agrega más puntos "malignos" en el extremo derecho, en el rango de tumores grandes (cerca de 10), y vuelve a ejecutar la regresión lineal.
#     - Ahora el modelo predice bien el tumor más grande, ¡pero el punto en x=3 se predice de forma incorrecta!
# - Para limpiar o reiniciar la gráfica, vuelve a ejecutar la celda que contiene el comando de la gráfica.

# In[5]:


w_in = np.zeros((1))
b_in = 0
plt.close('all') 
addpt = plt_one_addpt_onclick( x_train,y_train, w_in, b_in, logistic=False)


# El ejemplo anterior muestra que el modelo lineal no es suficiente para modelar datos categóricos. El modelo se puede ampliar, como se describe en el siguiente laboratorio.

# ## ¡Felicidades!
# En este laboratorio:
# - Exploraste conjuntos de datos categóricos y cómo graficarlos.
# - Comprobaste que la regresión lineal no es suficiente para un problema de clasificación.

# In[ ]:





# In[ ]:
