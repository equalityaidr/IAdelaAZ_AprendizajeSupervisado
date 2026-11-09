#!/usr/bin/env python
# coding: utf-8

# # Laboratorio opcional: Regresión logística, frontera de decisión
# 

# ## Objetivos
# En este laboratorio vas a:
# - Graficar la frontera de decisión de un modelo de regresión logística. Esto te dará una mejor idea de lo que el modelo está prediciendo.
# 

# In[1]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'widget')
import matplotlib.pyplot as plt
from lab_utils_common import plot_data, sigmoid, draw_vthresh
plt.style.use('./deeplearning.mplstyle')


# ## Conjunto de datos
# 
# Supongamos que tienes el siguiente conjunto de datos de entrenamiento:
# - La variable de entrada `X` es un array de numpy con 6 ejemplos de entrenamiento, cada uno con dos variables
# - La variable de salida `y` también es un array de numpy con 6 ejemplos, y `y` vale `0` o `1`

# In[2]:


X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1) 


# ### Graficar los datos 
# 
# Usemos una función auxiliar para graficar estos datos. Los puntos con etiqueta $y=1$ se muestran como cruces rojas, y los puntos con etiqueta $y=0$, como círculos azules. 

# In[3]:


fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X, y, ax)

ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$')
ax.set_xlabel('$x_0$')
plt.show()


# ## Modelo de regresión logística
# 
# 
# * Supongamos que quieres entrenar con estos datos un modelo de regresión logística de la forma   
# 
#   $f(x) = g(w_0x_0+w_1x_1 + b)$
#   
#   donde $g(z) = \frac{1}{1+e^{-z}}$ es la función sigmoide
# 
# 
# * Digamos que entrenaste el modelo y obtuviste los parámetros $b = -3, w_0 = 1, w_1 = 1$. Es decir,
# 
#   $f(x) = g(x_0+x_1-3)$
# 
#   (Más adelante en el curso aprenderás a ajustar estos parámetros a los datos)
#   
#   
# Intentemos entender qué predice este modelo entrenado graficando su frontera de decisión.

# ### Repaso de regresión logística y frontera de decisión
# 
# * Recuerda que, en regresión logística, el modelo se representa como 
# 
#   $$f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = g(\mathbf{w} \cdot \mathbf{x}^{(i)} + b) \tag{1}$$
# 
#   donde $g(z)$ es la función sigmoide, que transforma cualquier valor de entrada en un valor entre 0 y 1:
# 
#   $g(z) = \frac{1}{1+e^{-z}}\tag{2}$
#   y $\mathbf{w} \cdot \mathbf{x}$ es el producto punto entre vectores:
#   
#   $$\mathbf{w} \cdot \mathbf{x} = w_0 x_0 + w_1 x_1$$
#   
#   
# * Interpretamos la salida del modelo ($f_{\mathbf{w},b}(x)$) como la probabilidad de que $y=1$ dado $\mathbf{x}$, con los parámetros $\mathbf{w}$ y $b$.
# * Por lo tanto, para obtener una predicción final ($y=0$ o $y=1$) del modelo de regresión logística, podemos usar la siguiente regla:
# 
#   si $f_{\mathbf{w},b}(x) >= 0.5$, predecir $y=1$
#   
#   si $f_{\mathbf{w},b}(x) < 0.5$, predecir $y=0$
#   
#   
# * Grafiquemos la función sigmoide para ver dónde $g(z) >= 0.5$

# In[ ]:


# Graficar sigmoid(z) para valores de -10 a 10
z = np.arange(-10,11)

fig,ax = plt.subplots(1,1,figsize=(5,3))
# Graficar z vs. sigmoid(z)
ax.plot(z, sigmoid(z), c="b")

ax.set_title("Función sigmoide")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax,0)


# * Como puedes ver, $g(z) >= 0.5$ cuando $z >=0$
# 
# * En un modelo de regresión logística, $z = \mathbf{w} \cdot \mathbf{x} + b$. Por lo tanto,
# 
#   si $\mathbf{w} \cdot \mathbf{x} + b >= 0$, el modelo predice $y=1$
#   
#   si $\mathbf{w} \cdot \mathbf{x} + b < 0$, el modelo predice $y=0$
#   
#   
#   
# ### Graficar la frontera de decisión
# 
# Volvamos a nuestro ejemplo para entender cómo hace predicciones el modelo de regresión logística.
# 
# * Nuestro modelo de regresión logística tiene la forma
# 
#   $f(\mathbf{x}) = g(-3 + x_0+x_1)$
# 
# 
# * Por lo que vimos arriba, este modelo predice $y=1$ si $-3 + x_0+x_1 >= 0$
# 
# Veamos cómo se ve esto gráficamente. Empezaremos graficando $-3 + x_0+x_1 = 0$, que equivale a $x_1 = 3 - x_0$.
# 

# In[ ]:


# Elegir valores entre 0 y 6
x0 = np.arange(0,6)

x1 = 3 - x0
fig,ax = plt.subplots(1,1,figsize=(5,4))
# Graficar la frontera de decisión
ax.plot(x0,x1, c="b")
ax.axis([0, 4, 0, 3.5])

# Sombrear la región debajo de la línea
ax.fill_between(x0,x1, alpha=0.2)

# Graficar los datos originales
plot_data(X,y,ax)
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
plt.show()


# * En la gráfica, la línea azul representa la recta $x_0 + x_1 - 3 = 0$, que corta el eje x1 en 3 (si $x_0$ = 0, entonces $x_1$ = 3) y el eje x0 en 3 (si $x_1$ = 0, entonces $x_0$ = 3). 
# 
# 
# * La región sombreada representa $-3 + x_0+x_1 < 0$. La región por encima de la línea es $-3 + x_0+x_1 > 0$.
# 
# 
# * Cualquier punto en la región sombreada (debajo de la línea) se clasifica como $y=0$. Cualquier punto sobre la línea o por encima de ella se clasifica como $y=1$. Esta línea se conoce como "frontera de decisión".
# 
# Como vimos en clase, usando términos polinomiales de mayor orden (por ejemplo, $f(x) = g( x_0^2 + x_1 -1)$), podemos obtener fronteras no lineales más complejas.

# ## ¡Felicidades!
# Exploraste la frontera de decisión en el contexto de la regresión logística.

# In[ ]:





# In[ ]:
