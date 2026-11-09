#!/usr/bin/env python
# coding: utf-8

# # Laboratorio opcional: Regresión lineal con múltiples variables
# 
# En este laboratorio vas a ampliar las estructuras de datos y las funciones desarrolladas antes para que admitan múltiples variables de entrada (features). Se actualizan varias funciones, lo que hace que el laboratorio parezca largo, pero los cambios sobre las versiones anteriores son pequeños y se revisan rápido.
# # Contenido
# - [&nbsp;&nbsp;1.1 Objetivos](#toc_15456_1.1)
# - [&nbsp;&nbsp;1.2 Herramientas](#toc_15456_1.2)
# - [&nbsp;&nbsp;1.3 Notación](#toc_15456_1.3)
# - [2 Planteamiento del problema](#toc_15456_2)
# - [&nbsp;&nbsp;2.1 Matriz X con nuestros ejemplos](#toc_15456_2.1)
# - [&nbsp;&nbsp;2.2 Vector de parámetros w, b](#toc_15456_2.2)
# - [3 Predicción del modelo con múltiples variables](#toc_15456_3)
# - [&nbsp;&nbsp;3.1 Predicción individual, elemento a elemento](#toc_15456_3.1)
# - [&nbsp;&nbsp;3.2 Predicción individual, vectorizada](#toc_15456_3.2)
# - [4 Cálculo del costo con múltiples variables](#toc_15456_4)
# - [5 Descenso de gradiente con múltiples variables](#toc_15456_5)
# - [&nbsp;&nbsp;5.1 Cálculo del gradiente con múltiples variables](#toc_15456_5.1)
# - [&nbsp;&nbsp;5.2 Descenso de gradiente con múltiples variables](#toc_15456_5.2)
# - [6 ¡Felicidades!](#toc_15456_6)
# 

# <a name="toc_15456_1.1"></a>
# ## 1.1 Objetivos
# - Ampliar las funciones de nuestro modelo de regresión para que admitan múltiples variables
#     - Ampliar las estructuras de datos para admitir múltiples variables
#     - Reescribir las funciones de predicción, costo y gradiente para admitir múltiples variables
#     - Usar `np.dot` de NumPy para vectorizar las implementaciones y hacerlas más rápidas y simples

# <a name="toc_15456_1.2"></a>
# ## 1.2 Herramientas
# En este laboratorio usaremos: 
# - NumPy, una librería popular de computación científica
# - Matplotlib, una librería popular de visualización de datos

# In[ ]:


import copy, math
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('./deeplearning.mplstyle')
np.set_printoptions(precision=2)  # reduce la precisión con que se muestran los arrays de numpy


# <a name="toc_15456_1.3"></a>
# ## 1.3 Notación
# Aquí tienes un resumen de la notación que encontrarás, actualizada para múltiples variables.  
# 
# |General <img width=70/> <br />  Notación  <img width=70/> | Descripción<img width=350/>| Python (si aplica) |
# |: ------------|: ------------------------------------------------------------||
# | $a$ | escalar, sin negrita                                                      ||
# | $\mathbf{a}$ | vector, en negrita                                                 ||
# | $\mathbf{A}$ | matriz, mayúscula en negrita                                         ||
# | **Regresión** |         |    |     |
# |  $\mathbf{X}$ | matriz de ejemplos de entrenamiento                  | `X_train` |   
# |  $\mathbf{y}$  | valores objetivo de los ejemplos de entrenamiento                | `y_train` 
# |  $\mathbf{x}^{(i)}$, $y^{(i)}$ | $i$-ésimo ejemplo de entrenamiento | `X[i]`, `y[i]`|
# | m | número de ejemplos de entrenamiento | `m`|
# | n | número de variables en cada ejemplo | `n`|
# |  $\mathbf{w}$  |  parámetro: pesos                       | `w`    |
# |  $b$           |  parámetro: sesgo                                           | `b`    |     
# | $f_{\mathbf{w},b}(\mathbf{x}^{(i)})$ | Resultado de evaluar el modelo en $\mathbf{x^{(i)}}$, parametrizado por $\mathbf{w},b$: $f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w} \cdot \mathbf{x}^{(i)}+b$  | `f_wb` | 
# 

# <a name="toc_15456_2"></a>
# # 2 Planteamiento del problema
# 
# Usarás el ejemplo de predicción del precio de viviendas. El conjunto de entrenamiento tiene tres ejemplos con cuatro variables (tamaño, habitaciones, pisos y antigüedad), como se muestra en la tabla. Observa que, a diferencia de los laboratorios anteriores, el tamaño está en pies² y no en miles de pies². Esto causa un problema, ¡que resolverás en el próximo laboratorio!
# 
# | Tamaño (pies²) | Número de habitaciones  | Número de pisos | Antigüedad de la casa | Precio (miles de dólares)  |   
# | ----------------| ------------------- |----------------- |--------------|-------------- |  
# | 2104            | 5                   | 1                | 45           | 460           |  
# | 1416            | 3                   | 2                | 40           | 232           |  
# | 852             | 2                   | 1                | 35           | 178           |  
# 
# Vas a construir un modelo de regresión lineal con estos valores para luego predecir el precio de otras casas; por ejemplo, una casa de 1200 pies², 3 habitaciones, 1 piso y 40 años de antigüedad.  
# 
# Ejecuta la siguiente celda de código para crear tus variables `X_train` y `y_train`.

# In[ ]:


X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])


# <a name="toc_15456_2.1"></a>
# ## 2.1 Matriz X con nuestros ejemplos
# Igual que en la tabla, los ejemplos se guardan en una matriz de NumPy, `X_train`. Cada fila de la matriz representa un ejemplo. Cuando tienes $m$ ejemplos de entrenamiento ($m$ es tres en nuestro caso) y $n$ variables (cuatro en nuestro caso), $\mathbf{X}$ es una matriz de dimensiones ($m$, $n$) (m filas, n columnas).
# 
# 
# $$\mathbf{X} = 
# \begin{pmatrix}
#  x^{(0)}_0 & x^{(0)}_1 & \cdots & x^{(0)}_{n-1} \\ 
#  x^{(1)}_0 & x^{(1)}_1 & \cdots & x^{(1)}_{n-1} \\
#  \cdots \\
#  x^{(m-1)}_0 & x^{(m-1)}_1 & \cdots & x^{(m-1)}_{n-1} 
# \end{pmatrix}
# $$
# notación:
# - $\mathbf{x}^{(i)}$ es el vector que contiene el ejemplo i. $\mathbf{x}^{(i)}$ $ = (x^{(i)}_0, x^{(i)}_1, \cdots,x^{(i)}_{n-1})$
# - $x^{(i)}_j$ es el elemento j del ejemplo i. El superíndice entre paréntesis indica el número de ejemplo, mientras que el subíndice indica el elemento.  
# 
# Muestra los datos de entrada.

# In[ ]:


# los datos se guardan en un array/matriz de numpy
print(f"Forma de X: {X_train.shape}, Tipo de X: {type(X_train)}")
print(X_train)
print(f"Forma de y: {y_train.shape}, Tipo de y: {type(y_train)}")
print(y_train)


# <a name="toc_15456_2.2"></a>
# ## 2.2 Vector de parámetros w, b
# 
# * $\mathbf{w}$ es un vector con $n$ elementos.
#   - Cada elemento contiene el parámetro asociado a una variable.
#   - En nuestro conjunto de datos, n es 4.
#   - Conceptualmente, lo representamos como un vector columna
# 
# $$\mathbf{w} = \begin{pmatrix}
# w_0 \\ 
# w_1 \\
# \cdots\\
# w_{n-1}
# \end{pmatrix}
# $$
# * $b$ es un parámetro escalar.  

# Para la demostración, $\mathbf{w}$ y $b$ se cargarán con valores iniciales cercanos a los óptimos. $\mathbf{w}$ es un vector 1-D de NumPy.

# In[ ]:


b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"Forma de w_init: {w_init.shape}, tipo de b_init: {type(b_init)}")


# <a name="toc_15456_3"></a>
# # 3 Predicción del modelo con múltiples variables
# La predicción del modelo con múltiples variables está dada por el modelo lineal:
# 
# $$ f_{\mathbf{w},b}(\mathbf{x}) =  w_0x_0 + w_1x_1 +... + w_{n-1}x_{n-1} + b \tag{1}$$
# o, en notación vectorial:
# $$ f_{\mathbf{w},b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b  \tag{2} $$ 
# donde $\cdot$ es el `producto punto` entre vectores
# 
# Para mostrar el producto punto, implementaremos la predicción usando (1) y (2).

# <a name="toc_15456_3.1"></a>
# ## 3.1 Predicción individual, elemento a elemento
# Nuestra predicción anterior multiplicaba el valor de una variable por un parámetro y le sumaba el sesgo. Extender esa implementación a múltiples variables de forma directa significa implementar (1) con un ciclo que recorra cada elemento, lo multiplique por su parámetro y, al final, sume el sesgo.
# 

# In[ ]:


def predict_single_loop(x, w, b): 
    """
    predicción individual con regresión lineal
    
    Args:
      x (ndarray): Forma (n,), ejemplo con múltiples variables
      w (ndarray): Forma (n,), parámetros del modelo    
      b (scalar):  parámetro del modelo     
      
    Returns:
      p (scalar):  predicción
    """
    n = x.shape[0]
    p = 0
    for i in range(n):
        p_i = x[i] * w[i]  
        p = p + p_i         
    p = p + b                
    return p


# In[ ]:


# tomar una fila de los datos de entrenamiento
x_vec = X_train[0,:]
print(f"Forma de x_vec: {x_vec.shape}, valor de x_vec: {x_vec}")

# hacer una predicción
f_wb = predict_single_loop(x_vec, w_init, b_init)
print(f"Forma de f_wb: {f_wb.shape}, predicción: {f_wb}")


# Observa la forma de `x_vec`: es un vector 1-D de NumPy con 4 elementos, (4,). El resultado, `f_wb`, es un escalar.

# <a name="toc_15456_3.2"></a>
# ## 3.2 Predicción individual, vectorizada
# 
# La ecuación (1) se puede implementar con el producto punto, como en (2). Así podemos usar operaciones vectoriales para acelerar las predicciones.
# 
# Recuerda del laboratorio de Python/NumPy que `np.dot()` [[enlace](https://numpy.org/doc/stable/reference/generated/numpy.dot.html)] sirve para calcular el producto punto entre vectores. 

# In[ ]:


def predict(x, w, b): 
    """
    predicción individual con regresión lineal
    Args:
      x (ndarray): Forma (n,), ejemplo con múltiples variables
      w (ndarray): Forma (n,), parámetros del modelo   
      b (scalar):             parámetro del modelo 
      
    Returns:
      p (scalar):  predicción
    """
    p = np.dot(x, w) + b     
    return p    


# In[ ]:


# tomar una fila de los datos de entrenamiento
x_vec = X_train[0,:]
print(f"Forma de x_vec: {x_vec.shape}, valor de x_vec: {x_vec}")

# hacer una predicción
f_wb = predict(x_vec,w_init, b_init)
print(f"Forma de f_wb: {f_wb.shape}, predicción: {f_wb}")


# Los resultados y las formas son los mismos que en la versión anterior con ciclo. De aquí en adelante usaremos `np.dot` para estas operaciones. Ahora la predicción es una sola instrucción, así que la mayoría de las funciones la implementarán directamente en lugar de llamar a una función de predicción aparte.

# <a name="toc_15456_4"></a>
# # 4 Cálculo del costo con múltiples variables
# La ecuación de la función de costo con múltiples variables, $J(\mathbf{w},b)$, es:
# $$J(\mathbf{w},b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})^2 \tag{3}$$ 
# donde:
# $$ f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w} \cdot \mathbf{x}^{(i)} + b  \tag{4} $$ 
# 
# 
# A diferencia de los laboratorios anteriores, $\mathbf{w}$ y $\mathbf{x}^{(i)}$ son vectores y no escalares, para admitir múltiples variables.

# Abajo hay una implementación de las ecuaciones (3) y (4). Observa que sigue un *patrón estándar de este curso*: un ciclo for que recorre los `m` ejemplos.

# In[ ]:


def compute_cost(X, y, w, b): 
    """
    calcula el costo
    Args:
      X (ndarray (m,n)): Datos, m ejemplos con n variables
      y (ndarray (m,)) : valores objetivo
      w (ndarray (n,)) : parámetros del modelo  
      b (scalar)       : parámetro del modelo
      
    Returns:
      cost (scalar): costo
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b           #(n,)(n,) = escalar (ver np.dot)
        cost = cost + (f_wb_i - y[i])**2       #escalar
    cost = cost / (2 * m)                      #escalar    
    return cost


# In[ ]:


# Calcular y mostrar el costo usando los parámetros óptimos preseleccionados. 
cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'Costo con w óptimo: {cost}')


# **Resultado esperado**: Costo con w óptimo: 1.5578904045996674e-12

# <a name="toc_15456_5"></a>
# # 5 Descenso de gradiente con múltiples variables
# Descenso de gradiente para múltiples variables:
# 
# $$\begin{align*} \text{repetir}&\text{ hasta converger:} \; \lbrace \newline\;
# & w_j = w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} \tag{5}  \; & \text{para j = 0..n-1}\newline
# &b\ \ = b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b}  \newline \rbrace
# \end{align*}$$
# 
# donde n es el número de variables, los parámetros $w_j$ y $b$ se actualizan simultáneamente, y donde  
# 
# $$
# \begin{align}
# \frac{\partial J(\mathbf{w},b)}{\partial w_j}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)} \tag{6}  \\
# \frac{\partial J(\mathbf{w},b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) \tag{7}
# \end{align}
# $$
# * m es el número de ejemplos de entrenamiento del conjunto de datos
# 
#     
# * $f_{\mathbf{w},b}(\mathbf{x}^{(i)})$ es la predicción del modelo, mientras que $y^{(i)}$ es el valor objetivo
# 

# <a name="toc_15456_5.1"></a>
# ## 5.1 Cálculo del gradiente con múltiples variables
# Abajo hay una implementación para calcular las ecuaciones (6) y (7). Hay muchas formas de implementarlo. En esta versión hay
# - un ciclo externo que recorre los m ejemplos. 
#     - $\frac{\partial J(\mathbf{w},b)}{\partial b}$ para cada ejemplo se calcula directamente y se acumula
#     - en un segundo ciclo que recorre las n variables:
#         - se calcula $\frac{\partial J(\mathbf{w},b)}{\partial w_j}$ para cada $w_j$.
#    

# In[ ]:


def compute_gradient(X, y, w, b): 
    """
    Calcula el gradiente para regresión lineal 
    Args:
      X (ndarray (m,n)): Datos, m ejemplos con n variables
      y (ndarray (m,)) : valores objetivo
      w (ndarray (n,)) : parámetros del modelo  
      b (scalar)       : parámetro del modelo
      
    Returns:
      dj_dw (ndarray (n,)): El gradiente del costo con respecto a los parámetros w. 
      dj_db (scalar):       El gradiente del costo con respecto al parámetro b. 
    """
    m,n = X.shape           #(número de ejemplos, número de variables)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw


# In[ ]:


#Calcular y mostrar el gradiente 
tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db con w,b iniciales: {tmp_dj_db}')
print(f'dj_dw con w,b iniciales: \n {tmp_dj_dw}')


# **Resultado esperado**:   
# dj_db con w,b iniciales: -1.6739251122999121e-06  
# dj_dw con w,b iniciales:   
#  [-2.73e-03 -6.27e-06 -2.22e-06 -6.92e-05]  

# <a name="toc_15456_5.2"></a>
# ## 5.2 Descenso de gradiente con múltiples variables
# La siguiente función implementa la ecuación (5).

# In[ ]:


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Ejecuta descenso de gradiente por lotes (batch) para aprender w y b. Actualiza w y b dando 
    num_iters pasos de gradiente con tasa de aprendizaje alpha
    
    Args:
      X (ndarray (m,n))   : Datos, m ejemplos con n variables
      y (ndarray (m,))    : valores objetivo
      w_in (ndarray (n,)) : parámetros iniciales del modelo  
      b_in (scalar)       : parámetro inicial del modelo
      cost_function       : función para calcular el costo
      gradient_function   : función para calcular el gradiente
      alpha (float)       : Tasa de aprendizaje
      num_iters (int)     : número de iteraciones del descenso de gradiente
      
    Returns:
      w (ndarray (n,)) : Valores actualizados de los parámetros 
      b (scalar)       : Valor actualizado del parámetro 
      """
    
    # Lista para guardar el costo J en cada iteración, principalmente para graficar después
    J_history = []
    w = copy.deepcopy(w_in)  #evita modificar el w global dentro de la función
    b = b_in
    
    for i in range(num_iters):

        # Calcular el gradiente
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Actualizar los parámetros usando w, b, alpha y el gradiente
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Guardar el costo J en cada iteración
        if i<100000:      # evita agotar los recursos 
            J_history.append( cost_function(X, y, w, b))

        # Imprimir el costo 10 veces a lo largo del proceso (o en cada iteración si son menos de 10)
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteración {i:4d}: Costo {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #devuelve w, b finales y el historial de J para graficar


# En la siguiente celda vas a probar la implementación. 

# In[ ]:


# inicializar los parámetros
initial_w = np.zeros_like(w_init)
initial_b = 0.
# configuración del descenso de gradiente
iterations = 1000
alpha = 5.0e-7
# ejecutar el descenso de gradiente 
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)
print(f"b,w encontrados por descenso de gradiente: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
for i in range(m):
    print(f"predicción: {np.dot(X_train[i], w_final) + b_final:0.2f}, valor objetivo: {y_train[i]}")


# **Resultado esperado**:    
# b,w encontrados por descenso de gradiente: -0.00,[ 0.2   0.   -0.01 -0.07]   
# predicción: 426.19, valor objetivo: 460  
# predicción: 286.17, valor objetivo: 232  
# predicción: 171.47, valor objetivo: 178  

# In[ ]:


# graficar el costo vs. la iteración  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Costo vs. iteración");  ax2.set_title("Costo vs. iteración (tramo final)")
ax1.set_ylabel('Costo')             ;  ax2.set_ylabel('Costo') 
ax1.set_xlabel('Iteración')   ;  ax2.set_xlabel('Iteración') 
plt.show()


# *¡Estos resultados no son muy alentadores!* El costo sigue bajando y nuestras predicciones no son muy precisas. En el próximo laboratorio exploraremos cómo mejorar esto.

# 
# <a name="toc_15456_6"></a>
# # 6 ¡Felicidades!
# En este laboratorio:
# - Volviste a desarrollar las funciones de regresión lineal, ahora con múltiples variables.
# - Usaste `np.dot` de NumPy para vectorizar las implementaciones.
