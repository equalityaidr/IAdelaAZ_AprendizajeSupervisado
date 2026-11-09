#!/usr/bin/env python
# coding: utf-8

# # Laboratorio opcional: Descenso de gradiente para regresión logística

# ## Objetivos
# En este laboratorio vas a:
# - Adaptar el descenso de gradiente a la regresión logística.
# - Explorar el descenso de gradiente con un conjunto de datos que ya conoces.

# In[ ]:


import copy, math
import numpy as np
get_ipython().run_line_magic('matplotlib', 'widget')
import matplotlib.pyplot as plt
from lab_utils_common import  dlc, plot_data, plt_tumor_data, sigmoid, compute_cost_logistic
from plt_quad_logistic import plt_quad_logistic, plt_prob
plt.style.use('./deeplearning.mplstyle')


# ## Conjunto de datos 
# Empecemos con el mismo conjunto de datos de dos variables que usamos en el laboratorio de frontera de decisión.

# In[ ]:


X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])


# Como antes, usaremos una función auxiliar para graficar estos datos. Los puntos con etiqueta $y=1$ se muestran como cruces rojas, y los puntos con etiqueta $y=0$, como círculos azules.

# In[ ]:


fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X_train, y_train, ax)

ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.show()


# ## Descenso de gradiente logístico
# <img align="right" src="./images/C1_W3_Logistic_gradient_descent.png"     style=" width:400px; padding: 10px; " >
# 
# Recuerda que el algoritmo de descenso de gradiente usa el cálculo del gradiente:
# $$\begin{align*}
# &\text{repetir hasta converger:} \; \lbrace \\
# &  \; \; \;w_j = w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} \tag{1}  \; & \text{para j := 0..n-1} \\ 
# &  \; \; \;  \; \;b = b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b} \\
# &\rbrace
# \end{align*}$$
# 
# donde en cada iteración se actualizan simultáneamente los $w_j$ para todo $j$, y donde
# $$\begin{align*}
# \frac{\partial J(\mathbf{w},b)}{\partial w_j}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)} \tag{2} \\
# \frac{\partial J(\mathbf{w},b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) \tag{3} 
# \end{align*}$$
# 
# * m es el número de ejemplos de entrenamiento del conjunto de datos      
# * $f_{\mathbf{w},b}(x^{(i)})$ es la predicción del modelo, mientras que $y^{(i)}$ es el valor objetivo
# * En un modelo de regresión logística  
#     $z = \mathbf{w} \cdot \mathbf{x} + b$  
#     $f_{\mathbf{w},b}(x) = g(z)$  
#     donde $g(z)$ es la función sigmoide:  
#     $g(z) = \frac{1}{1+e^{-z}}$   
#     
# 

# ### Implementación del descenso de gradiente
# La implementación del algoritmo de descenso de gradiente tiene dos componentes: 
# - El ciclo que implementa la ecuación (1). Es la función `gradient_descent` de abajo, que normalmente se te da ya hecha en los laboratorios opcionales y de práctica.
# - El cálculo del gradiente actual, ecuaciones (2) y (3). Es la función `compute_gradient_logistic` de abajo. Se te pedirá implementarla en el laboratorio de práctica de esta semana.
# 
# #### Cálculo del gradiente: descripción del código
# Implementa las ecuaciones (2) y (3) para todos los $w_j$ y $b$.
# Hay muchas formas de implementarlo. Esta es la que se usa aquí:
# - inicializar las variables donde se acumulan `dj_dw` y `dj_db`
# - para cada ejemplo:
#     - calcular el error de ese ejemplo, $g(\mathbf{w} \cdot \mathbf{x}^{(i)} + b) - \mathbf{y}^{(i)}$
#     - para cada valor de entrada $x_{j}^{(i)}$ de ese ejemplo:  
#         - multiplicar el error por la entrada $x_{j}^{(i)}$ y sumarlo al elemento correspondiente de `dj_dw` (ecuación 2)
#     - sumar el error a `dj_db` (ecuación 3)
# 
# - dividir `dj_db` y `dj_dw` entre el número total de ejemplos (m)
# - observa que, en numpy, $\mathbf{x}^{(i)}$ es `X[i,:]` o `X[i]`, y $x_{j}^{(i)}$ es `X[i,j]`

# In[ ]:


def compute_gradient_logistic(X, y, w, b): 
    """
    Calcula el gradiente para regresión logística 
 
    Args:
      X (ndarray (m,n)): Datos, m ejemplos con n variables
      y (ndarray (m,)) : valores objetivo
      w (ndarray (n,)) : parámetros del modelo  
      b (scalar)       : parámetro del modelo
    Returns
      dj_dw (ndarray (n,)): El gradiente del costo con respecto a los parámetros w. 
      dj_db (scalar)      : El gradiente del costo con respecto al parámetro b. 
    """
    m,n = X.shape
    dj_dw = np.zeros((n,))                           #(n,)
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          #(n,)(n,)=escalar
        err_i  = f_wb_i  - y[i]                       #escalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      #escalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   #(n,)
    dj_db = dj_db/m                                   #escalar
        
    return dj_db, dj_dw  


# Verifica la implementación de la función de gradiente con la siguiente celda.

# In[ ]:


X_tmp = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_tmp = np.array([0, 0, 0, 1, 1, 1])
w_tmp = np.array([2.,3.])
b_tmp = 1.
dj_db_tmp, dj_dw_tmp = compute_gradient_logistic(X_tmp, y_tmp, w_tmp, b_tmp)
print(f"dj_db: {dj_db_tmp}" )
print(f"dj_dw: {dj_dw_tmp.tolist()}" )


# **Salida esperada**
# ``` 
# dj_db: 0.49861806546328574
# dj_dw: [0.498333393278696, 0.49883942983996693]
# ```

# #### Código del descenso de gradiente 
# Abajo está el código que implementa la ecuación (1). Tómate un momento para identificar las funciones dentro de la rutina y compararlas con las ecuaciones de arriba.

# In[ ]:


def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
    """
    Ejecuta descenso de gradiente por lotes (batch)
    
    Args:
      X (ndarray (m,n))  : Datos, m ejemplos con n variables
      y (ndarray (m,))   : valores objetivo
      w_in (ndarray (n,)): Valores iniciales de los parámetros del modelo  
      b_in (scalar)      : Valor inicial del parámetro del modelo
      alpha (float)      : Tasa de aprendizaje
      num_iters (scalar) : número de iteraciones del descenso de gradiente
      
    Returns:
      w (ndarray (n,))   : Valores actualizados de los parámetros
      b (scalar)         : Valor actualizado del parámetro 
    """
    # Lista para guardar el costo J en cada iteración, principalmente para graficar después
    J_history = []
    w = copy.deepcopy(w_in)  #evita modificar el w global dentro de la función
    b = b_in
    
    for i in range(num_iters):
        # Calcular el gradiente
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)   

        # Actualizar los parámetros usando w, b, alpha y el gradiente
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
        # Guardar el costo J en cada iteración
        if i<100000:      # evita agotar los recursos 
            J_history.append( compute_cost_logistic(X, y, w, b) )

        # Imprimir el costo 10 veces a lo largo del proceso (o en cada iteración si son menos de 10)
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteración {i:4d}: Costo {J_history[-1]}   ")
        
    return w, b, J_history         #devuelve w, b finales y el historial de J para graficar


# Ejecutemos el descenso de gradiente con nuestro conjunto de datos.

# In[ ]:


w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alph = 0.1
iters = 10000

w_out, b_out, _ = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters) 
print(f"\nparámetros actualizados: w:{w_out}, b:{b_out}")


# #### Grafiquemos los resultados del descenso de gradiente:

# In[ ]:


fig,ax = plt.subplots(1,1,figsize=(5,4))
# graficar la probabilidad 
plt_prob(ax, w_out, b_out)

# Graficar los datos originales
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')   
ax.axis([0, 4, 0, 3.5])
plot_data(X_train,y_train,ax)

# Graficar la frontera de decisión
x0 = -b_out/w_out[0]
x1 = -b_out/w_out[1]
ax.plot([0,x0],[x1,0], c=dlc["dlblue"], lw=1)
plt.show()


# En la gráfica:
#  - el sombreado refleja la probabilidad de que y=1 (el resultado antes de aplicar la frontera de decisión)
#  - la frontera de decisión es la línea donde la probabilidad = 0.5
#  

# ## Otro conjunto de datos
# Volvamos a un conjunto de datos de una sola variable. Con solo dos parámetros, $w$ y $b$, podemos graficar la función de costo con un gráfico de contorno para entender mejor qué está haciendo el descenso de gradiente.

# In[ ]:


x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])


# Como antes, usaremos una función auxiliar para graficar estos datos. Los puntos con etiqueta $y=1$ se muestran como cruces rojas, y los puntos con etiqueta $y=0$, como círculos azules.

# In[ ]:


fig,ax = plt.subplots(1,1,figsize=(4,3))
plt_tumor_data(x_train, y_train, ax)
plt.show()


# En la gráfica de abajo, prueba lo siguiente:
# - Cambia $w$ y $b$ haciendo clic dentro del gráfico de contorno de la parte superior derecha.
#     - Los cambios pueden tardar uno o dos segundos.
#     - Observa cómo cambia el valor del costo en la gráfica superior izquierda.
#     - Observa que el costo es la acumulación de la pérdida de cada ejemplo (líneas punteadas verticales).
# - Ejecuta el descenso de gradiente haciendo clic en el botón naranja.
#     - Observa cómo el costo disminuye de forma constante (el gráfico de contorno y la gráfica del costo están en log(costo)).
#     - Hacer clic en el gráfico de contorno reinicia el modelo para una nueva ejecución.
# - Para reiniciar la gráfica, vuelve a ejecutar la celda.

# In[ ]:


w_range = np.array([-1, 7])
b_range = np.array([1, -14])
quad = plt_quad_logistic( x_train, y_train, w_range, b_range )


# ## ¡Felicidades!
# En este laboratorio:
# - Examinaste las fórmulas y la implementación del cálculo del gradiente para regresión logística.
# - Usaste esas funciones para
#     - explorar un conjunto de datos de una variable
#     - explorar un conjunto de datos de dos variables

# In[ ]:
