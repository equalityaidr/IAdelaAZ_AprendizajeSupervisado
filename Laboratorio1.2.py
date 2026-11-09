#!/usr/bin/env python
# coding: utf-8

# # Laboratorio opcional: Función de costo 
# <figure>
#     <center> <img src="./images/C1_W1_L3_S2_Lecture_b.png"  style="width:1000px;height:200px;" ></center>
# </figure>
# 
# 

# ## Objetivos
# En este laboratorio vas a:
# - Implementar y explorar la función de `costo` para regresión lineal con una variable. 
# 

# ## Herramientas
# En este laboratorio usaremos: 
# - NumPy, una librería popular de computación científica
# - Matplotlib, una librería popular de visualización de datos
# - Funciones de graficación incluidas en el archivo lab_utils_uni.py, en el directorio local

# In[ ]:


import numpy as np
#get_ipython().run_line_magic('matplotlib', 'widget')
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
#plt.style.use('./deeplearning.mplstyle')


# ## Planteamiento del problema
# 
# Quieres un modelo que pueda predecir el precio de una vivienda a partir de su tamaño.  
# Usemos los mismos dos puntos de datos del laboratorio anterior: una casa de 1000 pies cuadrados vendida en \\$300,000 y una casa de 2000 pies cuadrados vendida en \\$500,000.
# 
# 
# | Tamaño (1000 pies²)     | Precio (miles de dólares) |
# | -------------------| ------------------------ |
# | 1                 | 300                      |
# | 2                  | 500                      |
# 

# In[ ]:


x_train = np.array([1.0, 2.0])           #(tamaño en miles de pies cuadrados)
y_train = np.array([300.0, 500.0])           #(precio en miles de dólares)


# ## Cálculo del costo
# El término "costo" puede prestarse a confusión en este ejercicio, ya que los datos son precios de viviendas. Aquí, el costo es una medida de qué tan bien nuestro modelo predice el precio objetivo de la casa. Para los datos de viviendas usamos el término "precio".
# 
# La ecuación del costo con una variable es:
#   $$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \tag{1}$$ 
#  
# donde 
#   $$f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{2}$$
#   
# - $f_{w,b}(x^{(i)})$ es nuestra predicción para el ejemplo $i$ usando los parámetros $w,b$.  
# - $(f_{w,b}(x^{(i)}) -y^{(i)})^2$ es la diferencia al cuadrado entre el valor objetivo y la predicción.   
# - Estas diferencias se suman para los $m$ ejemplos y se dividen entre `2m` para obtener el costo, $J(w,b)$.  
# >Nota: en clase, las sumatorias suelen ir de 1 a m, mientras que en el código van de 0 a m-1.
# 

# El código de abajo calcula el costo recorriendo cada ejemplo. En cada iteración:
# - se calcula `f_wb`, una predicción;
# - se calcula la diferencia entre el objetivo y la predicción, y se eleva al cuadrado;
# - ese valor se suma al costo total.

# In[ ]:


def compute_cost(x, y, w, b): 
    """
    Calcula la función de costo para regresión lineal.
    
    Args:
      x (ndarray (m,)): Datos, m ejemplos 
      y (ndarray (m,)): valores objetivo
      w,b (scalar)    : parámetros del modelo  
    
    Returns
        total_cost (float): El costo de usar w,b como parámetros de la regresión lineal
               para ajustar los puntos de datos en x e y
    """
    # número de ejemplos de entrenamiento
    m = x.shape[0] 
    
    cost_sum = 0 
    for i in range(m): 
        f_wb = w * x[i] + b   
        cost = (f_wb - y[i]) ** 2  
        cost_sum = cost_sum + cost  
    total_cost = (1 / (2 * m)) * cost_sum  

    return total_cost


# ## Intuición de la función de costo

# <img align="left" src="./images/C1_W1_Lab02_GoalOfRegression.PNG"    style=" width:380px; padding: 10px;  " /> Tu objetivo es encontrar un modelo $f_{w,b}(x) = wx + b$, con parámetros $w,b$, que prediga con precisión el valor de las casas dada una entrada $x$. El costo es una medida de qué tan preciso es el modelo con los datos de entrenamiento.
# 
# La ecuación de costo (1) muestra que, si se pueden elegir $w$ y $b$ de modo que las predicciones $f_{w,b}(x)$ coincidan con los datos objetivo $y$, el término $(f_{w,b}(x^{(i)}) - y^{(i)})^2 $ será cero y el costo quedará minimizado. ¡En este ejemplo sencillo de dos puntos, puedes lograrlo!
# 
# En el laboratorio anterior determinaste que $b=100$ daba una solución óptima, así que fijemos $b$ en 100 y concentrémonos en $w$.
# 
# <br/>
# Abajo, usa el control deslizante para seleccionar el valor de $w$ que minimiza el costo. La gráfica puede tardar unos segundos en actualizarse.

# In[ ]:


plt_intuition(x_train,y_train)


# La gráfica muestra algunos puntos que vale la pena destacar:
# - el costo se minimiza cuando $w = 200$, lo que coincide con los resultados del laboratorio anterior;
# - como la diferencia entre el objetivo y la predicción se eleva al cuadrado en la ecuación de costo, el costo aumenta rápidamente cuando $w$ es demasiado grande o demasiado pequeño;
# - usar los valores de `w` y `b` que minimizan el costo produce una línea que se ajusta perfectamente a los datos.

# ## Visualización de la función de costo en 3D
# 
# Puedes ver cómo varía el costo con respecto a *ambos* parámetros, `w` y `b`, graficándolo en 3D o con un gráfico de contorno.   
# Vale la pena mencionar que algunas gráficas de este curso pueden ser bastante elaboradas. Las funciones de graficación se proporcionan y, aunque puede ser instructivo leer el código para familiarizarse con los métodos, no es necesario para completar el curso con éxito. Estas funciones están en lab_utils_uni.py, en el directorio local.

# ### Conjunto de datos más grande
# Es instructivo observar un escenario con algunos puntos de datos más. Este conjunto incluye puntos que no caen sobre la misma línea. ¿Qué significa eso para la ecuación de costo? ¿Podemos encontrar $w$ y $b$ que den un costo de 0? 

# In[ ]:


x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,])


# En el gráfico de contorno, haz clic en un punto para seleccionar los valores de `w` y `b` que logren el menor costo. Usa los contornos como guía. Nota: la gráfica puede tardar unos segundos en actualizarse. 

# In[ ]:


plt.close('all') 
fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)


# Observa las líneas punteadas en la gráfica de la izquierda. Representan la porción del costo que aporta cada ejemplo del conjunto de entrenamiento. En este caso, valores aproximados de $w=209$ y $b=2.4$ producen un costo bajo. Como los ejemplos de entrenamiento no están sobre una línea, el costo mínimo no es cero.

# ### Superficie de costo convexa
# Como la función de costo eleva al cuadrado el error, la "superficie de error" es convexa, con forma de tazón de sopa. Siempre tendrá un mínimo que puede alcanzarse siguiendo el gradiente en todas las dimensiones. En la gráfica anterior esto no es fácil de reconocer, porque las dimensiones de $w$ y $b$ tienen escalas distintas. La siguiente gráfica, donde $w$ y $b$ son simétricos, se mostró en clase:

# In[ ]:


soup_bowl()


# # ¡Felicidades!
# Aprendiste lo siguiente:
#  - La ecuación de costo mide qué tan bien coinciden tus predicciones con los datos de entrenamiento.
#  - Minimizar el costo permite obtener los valores óptimos de $w$ y $b$.

# In[ ]:
