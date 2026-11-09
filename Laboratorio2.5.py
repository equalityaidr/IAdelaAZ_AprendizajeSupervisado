#!/usr/bin/env python
# coding: utf-8

# # Laboratorio opcional: Ingeniería de variables y regresión polinomial
# 
# ![](./images/C1_W2_Lab07_FeatureEngLecture.PNG)
# 

# ## Objetivos
# En este laboratorio vas a:
# - Explorar la ingeniería de variables (feature engineering) y la regresión polinomial, que te permiten usar las herramientas de la regresión lineal para ajustar funciones muy complicadas, incluso muy no lineales.
# 

# ## Herramientas
# Usarás las funciones desarrolladas en laboratorios anteriores, además de Matplotlib y NumPy. 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import zscore_normalize_features, run_gradient_descent_feng
np.set_printoptions(precision=2)  # reduce la precisión con que se muestran los arrays de numpy


# <a name='FeatureEng'></a>
# # Panorama de la ingeniería de variables y la regresión polinomial
# 
# Tal cual, la regresión lineal permite construir modelos de la forma:
# $$f_{\mathbf{w},b} = w_0x_0 + w_1x_1+ ... + w_{n-1}x_{n-1} + b \tag{1}$$ 
# ¿Qué pasa si tus variables o datos no son lineales, o son combinaciones de variables? Por ejemplo, el precio de las viviendas no suele ser lineal con respecto al área habitable, sino que penaliza las casas muy pequeñas o muy grandes, lo que produce curvas como las de la imagen de arriba. ¿Cómo podemos usar las herramientas de la regresión lineal para ajustar esa curva? Recuerda que la "herramienta" que tenemos es la capacidad de modificar los parámetros $\mathbf{w}$, $\mathbf{b}$ en (1) para "ajustar" la ecuación a los datos de entrenamiento. Sin embargo, por más que ajustemos $\mathbf{w}$,$\mathbf{b}$ en (1), no lograremos ajustar una curva no lineal.
# 

# <a name='PolynomialFeatures'></a>
# ## Variables polinomiales
# 
# Arriba consideramos un escenario donde los datos no eran lineales. Intentemos usar lo que sabemos hasta ahora para ajustar una curva no lineal. Empezaremos con una cuadrática simple: $y = 1+x^2$
# 
# Ya conoces todas las funciones que vamos a usar; están en el archivo lab_utils_multi.py, por si quieres revisarlas. Usaremos [`np.c_[..]`](https://numpy.org/doc/stable/reference/generated/numpy.c_.html), una función de NumPy que concatena arrays como columnas.

# In[ ]:


# crear los datos objetivo
x = np.arange(0, 20, 1)
y = 1 + x**2
X = x.reshape(-1, 1)

model_w,model_b = run_gradient_descent_feng(X,y,iterations=1000, alpha = 1e-2)

plt.scatter(x, y, marker='x', c='r', label="Valor real"); plt.title("sin ingeniería de variables")
plt.plot(x,X@model_w + model_b, label="Valor predicho");  plt.xlabel("X"); plt.ylabel("y"); plt.legend(); plt.show()


# Como era de esperar, el ajuste no es bueno. Lo que se necesita es algo como $y= w_0x_0^2 + b$, es decir, una **variable polinomial**.
# Para lograrlo, puedes modificar los *datos de entrada* para *construir* las variables necesarias. Si reemplazas los datos originales por una versión con el valor de $x$ al cuadrado, puedes obtener $y= w_0x_0^2 + b$. Probémoslo: abajo se reemplaza `X` por `X**2`.

# In[ ]:


# crear los datos objetivo
x = np.arange(0, 20, 1)
y = 1 + x**2

# Construir variables 
X = x**2      #<-- variable construida


# In[ ]:


X = X.reshape(-1, 1)  #X debe ser una matriz 2-D
model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha = 1e-5)

plt.scatter(x, y, marker='x', c='r', label="Valor real"); plt.title("Con la variable x**2")
plt.plot(x, np.dot(X,model_w) + model_b, label="Valor predicho"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()


# ¡Excelente! El ajuste es casi perfecto. Observa los valores de $\mathbf{w}$ y b que se imprimen justo encima de la gráfica: `w,b found by gradient descent: w: [1.], b: 0.0490`. El descenso de gradiente modificó los valores iniciales de $\mathbf{w},b$ hasta (1.0, 0.049), es decir, el modelo $y=1*x_0^2+0.049$, muy cercano a nuestro objetivo $y=1*x_0^2+1$. Si lo ejecutaras por más tiempo, el ajuste sería aún mejor. 

# ### Selección de variables
# <a name='GDF'></a>
# Arriba sabíamos que se necesitaba un término $x^2$. No siempre es obvio qué variables se necesitan. Se pueden agregar varias variables candidatas para tratar de encontrar las más útiles. Por ejemplo, ¿qué pasaría si hubiéramos probado con $y=w_0x_0 + w_1x_1^2 + w_2x_2^3+b$? 
# 
# Ejecuta las siguientes celdas. 

# In[ ]:


# crear los datos objetivo
x = np.arange(0, 20, 1)
y = x**2

# construir variables
X = np.c_[x, x**2, x**3]   #<-- variables construidas


# In[ ]:


model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha=1e-7)

plt.scatter(x, y, marker='x', c='r', label="Valor real"); plt.title("Variables x, x**2, x**3")
plt.plot(x, X@model_w + model_b, label="Valor predicho"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()


# Observa que el valor de $\mathbf{w}$ es `[0.08 0.54 0.03]` y el de b es `0.0106`. Esto significa que, después del ajuste/entrenamiento, el modelo es:
# $$ 0.08x + 0.54x^2 + 0.03x^3 + 0.0106 $$
# El descenso de gradiente le dio más peso a la variable que mejor se ajusta a los datos, $x^2$, aumentando el término $w_1$ en relación con los demás. Si lo ejecutaras durante mucho tiempo, seguiría reduciendo el efecto de los otros términos. 
# >El descenso de gradiente está eligiendo las variables "correctas" por nosotros, al darle más peso a su parámetro asociado
# 
# Repasemos esta idea:
# - Un peso menor indica una variable menos importante o menos adecuada. En el extremo, cuando el peso es cero o muy cercano a cero, la variable asociada no ayuda a ajustar el modelo a los datos.
# - Arriba, después del ajuste, el peso asociado a la variable $x^2$ es mucho mayor que los pesos de $x$ o $x^3$, porque es la más útil para ajustar los datos. 

# ### Otra forma de verlo
# Arriba, las variables polinomiales se eligieron según qué tan bien coincidían con los datos objetivo. Otra forma de pensarlo es notar que, una vez creadas las nuevas variables, seguimos usando regresión lineal. Por lo tanto, las mejores variables serán las que tengan una relación lineal con el objetivo. Se entiende mejor con un ejemplo. 

# In[ ]:


# crear los datos objetivo
x = np.arange(0, 20, 1)
y = x**2

# construir variables
X = np.c_[x, x**2, x**3]   #<-- variables construidas
X_features = ['x','x^2','x^3']


# In[ ]:


fig,ax=plt.subplots(1, 3, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X[:,i],y)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("y")
plt.show()


# Es claro que la variable $x^2$, graficada contra el valor objetivo $y$, tiene una relación lineal. Así, la regresión lineal puede generar fácilmente un modelo con esa variable.

# ### Escalado de variables
# Como vimos en el laboratorio anterior, si el conjunto de datos tiene variables con escalas muy distintas, conviene aplicar escalado de variables para acelerar el descenso de gradiente. En el ejemplo anterior tenemos $x$, $x^2$ y $x^3$, que naturalmente tienen escalas muy diferentes. Apliquemos normalización z-score a nuestro ejemplo.

# In[ ]:


# crear los datos objetivo
x = np.arange(0,20,1)
X = np.c_[x, x**2, x**3]
print(f"Rango (máx - mín) por columna en X original:    {np.ptp(X,axis=0)}")

# aplicar normalización z-score 
X = zscore_normalize_features(X)     
print(f"Rango (máx - mín) por columna en X normalizado: {np.ptp(X,axis=0)}")


# Ahora podemos intentarlo de nuevo con un valor de alfa más agresivo:

# In[ ]:


x = np.arange(0,20,1)
y = x**2

X = np.c_[x, x**2, x**3]
X = zscore_normalize_features(X) 

model_w, model_b = run_gradient_descent_feng(X, y, iterations=100000, alpha=1e-1)

plt.scatter(x, y, marker='x', c='r', label="Valor real"); plt.title("Variables x, x**2, x**3 normalizadas")
plt.plot(x,X@model_w + model_b, label="Valor predicho"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()


# El escalado de variables hace que converja mucho más rápido.   
# Observa de nuevo los valores de $\mathbf{w}$. El término $w_1$, que corresponde a $x^2$, es el que tiene más peso. El descenso de gradiente prácticamente eliminó el término $x^3$.

# ### Funciones complejas
# Con ingeniería de variables se pueden modelar incluso funciones bastante complejas:

# In[ ]:


x = np.arange(0,20,1)
y = np.cos(x/2)

X = np.c_[x, x**2, x**3,x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
X = zscore_normalize_features(X) 

model_w,model_b = run_gradient_descent_feng(X, y, iterations=1000000, alpha = 1e-1)

plt.scatter(x, y, marker='x', c='r', label="Valor real"); plt.title("Variables x hasta x**13 normalizadas")
plt.plot(x,X@model_w + model_b, label="Valor predicho"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()


# 
# ## ¡Felicidades!
# En este laboratorio:
# - Aprendiste cómo la regresión lineal puede modelar funciones complejas, incluso muy no lineales, usando ingeniería de variables.
# - Reconociste la importancia de aplicar escalado de variables al hacer ingeniería de variables.

# In[ ]:
