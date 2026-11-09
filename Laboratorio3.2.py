#!/usr/bin/env python
# coding: utf-8

# # Laboratorio opcional: Regresión logística
# 
# En este laboratorio sin calificación vas a: 
# - Explorar la función sigmoide (también conocida como función logística)
# - Explorar la regresión logística, que usa la función sigmoide

# In[ ]:


import numpy as np
#get_ipython().run_line_magic('matplotlib', 'widget')
import matplotlib.pyplot as plt
from plt_one_addpt_onclick import plt_one_addpt_onclick
from lab_utils_common import draw_vthresh
#plt.style.use('./deeplearning.mplstyle')


# ## Función sigmoide o logística
# <img align="left" src="./images/C1_W3_LogisticRegression_left.png"     style=" width:300px; padding: 10px; " >Como se vio en los videos de clase, para una tarea de clasificación podemos empezar usando nuestro modelo de regresión lineal, $f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w} \cdot  \mathbf{x}^{(i)} + b$, para predecir $y$ a partir de $x$. 
# - Sin embargo, queremos que las predicciones de nuestro modelo de clasificación estén entre 0 y 1, ya que la variable de salida $y$ solo puede ser 0 o 1. 
# - Esto se logra con una "función sigmoide", que transforma cualquier valor de entrada en un valor entre 0 y 1. 
# 
# 
# Implementemos la función sigmoide y comprobémoslo.
# 
# ## Fórmula de la función sigmoide
# 
# La fórmula de la función sigmoide es la siguiente:  
# 
# $g(z) = \frac{1}{1+e^{-z}}\tag{1}$
# 
# En regresión logística, z (la entrada de la función sigmoide) es la salida de un modelo de regresión lineal. 
# - Con un solo ejemplo, $z$ es un escalar.
# - Con varios ejemplos, $z$ puede ser un vector de $m$ valores, uno por cada ejemplo. 
# - La implementación de la función sigmoide debe funcionar con ambos formatos de entrada.
# Implementémosla en Python.

# NumPy tiene una función llamada [`exp()`](https://numpy.org/doc/stable/reference/generated/numpy.exp.html), que permite calcular fácilmente la exponencial ($e^{z}$) de todos los elementos del array de entrada (`z`).
#  
# También funciona con un solo número como entrada, como se muestra abajo.

# In[ ]:


# La entrada es un array. 
input_array = np.array([1,2,3])
exp_array = np.exp(input_array)

print("Entrada de exp:", input_array)
print("Salida de exp:", exp_array)

# La entrada es un solo número
input_val = 1  
exp_val = np.exp(input_val)

print("Entrada de exp:", input_val)
print("Salida de exp:", exp_val)


# La función `sigmoid` se implementa en Python como se muestra en la siguiente celda.

# In[ ]:


def sigmoid(z):
    """
    Calcula la sigmoide de z

    Args:
        z (ndarray): Un escalar o un array de numpy de cualquier tamaño.

    Returns:
        g (ndarray): sigmoid(z), con la misma forma que z
         
    """

    g = 1/(1+np.exp(-z))
   
    return g


# Veamos qué devuelve esta función para distintos valores de `z`.

# In[ ]:


# Generar un array de valores enteros consecutivos entre -10 y 10
z_tmp = np.arange(-10,11)

# Usar la función implementada arriba para obtener los valores de la sigmoide
y = sigmoid(z_tmp)

# Código para mostrar los dos arrays lado a lado
np.set_printoptions(precision=3) 
print("Entrada (z), Salida (sigmoid(z))")
print(np.c_[z_tmp, y])


# Los valores de la columna izquierda son `z` y los de la columna derecha son `sigmoid(z)`. Como puedes ver, los valores de entrada van de -10 a 10, y los de salida, de 0 a 1. 
# 
# Ahora grafiquemos esta función con la librería `matplotlib`.

# In[ ]:


# Graficar z vs. sigmoid(z)
fig,ax = plt.subplots(1,1,figsize=(5,3))
ax.plot(z_tmp, y, c="b")

ax.set_title("Función sigmoide")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax,0)


# Como puedes ver, la función sigmoide se acerca a `0` cuando `z` toma valores muy negativos, y se acerca a `1` cuando `z` toma valores muy positivos.
# 

# ## Regresión logística
# <img align="left" src="./images/C1_W3_LogisticRegression_right.png"     style=" width:300px; padding: 10px; " > Un modelo de regresión logística aplica la sigmoide al modelo de regresión lineal que ya conoces, así:
# 
# $$ f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = g(\mathbf{w} \cdot \mathbf{x}^{(i)} + b ) \tag{2} $$ 
# 
#   donde
# 
#   $g(z) = \frac{1}{1+e^{-z}}\tag{3}$
# 

#   
# Apliquemos la regresión logística al ejemplo de datos categóricos de clasificación de tumores.  
# Primero, carga los ejemplos y los valores iniciales de los parámetros.
#   
# 

# In[ ]:


x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

w_in = np.zeros((1))
b_in = 0


# Prueba los siguientes pasos:
# - Haz clic en "Run Logistic Regression" (ejecutar regresión logística) para encontrar el mejor modelo de regresión logística para los datos de entrenamiento.
#     - Observa que el modelo resultante se ajusta bastante bien a los datos.
#     - Observa que la línea naranja es "$z$", es decir, $\mathbf{w} \cdot \mathbf{x}^{(i)} + b$. No coincide con la línea de un modelo de regresión lineal.
# Puedes mejorar aún más estos resultados aplicando un *umbral*. 
# - Marca la casilla "Toggle 0.5 threshold" (activar umbral de 0.5) para ver las predicciones al aplicar el umbral.
#     - Estas predicciones se ven bien: coinciden con los datos.
#     - Ahora agrega más puntos en el rango de tumores grandes (cerca de 10) y vuelve a ejecutar la regresión logística.
#     - A diferencia del modelo de regresión lineal, este modelo sigue haciendo predicciones correctas.

# In[ ]:


plt.close('all') 
addpt = plt_one_addpt_onclick( x_train,y_train, w_in, b_in, logistic=True)


# ## ¡Felicidades!
# Exploraste el uso de la función sigmoide en la regresión logística.

# In[ ]:
