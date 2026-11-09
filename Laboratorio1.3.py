#!/usr/bin/env python
# coding: utf-8

# # Laboratorio opcional: Descenso de gradiente para regresión lineal
# 
# <figure>
#     <center> <img src="./images/C1_W1_L4_S1_Lecture_GD.png"  style="width:800px;height:200px;" ></center>
# </figure>

# ## Objetivos
# En este laboratorio vas a:
# - Automatizar el proceso de optimizar $w$ y $b$ usando descenso de gradiente.

# ## Herramientas
# En este laboratorio usaremos: 
# - NumPy, una librería popular de computación científica
# - Matplotlib, una librería popular de visualización de datos
# - Funciones de graficación incluidas en el archivo lab_utils_uni.py, en el directorio local

# In[ ]:


import math, copy
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('./deeplearning.mplstyle')
from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients


# <a name="toc_40291_2"></a>
# # Planteamiento del problema
# 
# Usemos los mismos dos puntos de datos de antes: una casa de 1000 pies cuadrados vendida en \\$300,000 y una casa de 2000 pies cuadrados vendida en \\$500,000.
# 
# | Tamaño (1000 pies²)     | Precio (miles de dólares) |
# | ----------------| ------------------------ |
# | 1               | 300                      |
# | 2               | 500                      |
# 

# In[ ]:


# Cargar el conjunto de datos
x_train = np.array([1.0, 2.0])   #variable de entrada (feature)
y_train = np.array([300.0, 500.0])   #valor objetivo


# <a name="toc_40291_2.0.1"></a>
# ### Compute_Cost
# Esta función la desarrollamos en el laboratorio anterior. La necesitaremos de nuevo aquí.

# In[ ]:


#Función para calcular el costo
def compute_cost(x, y, w, b):
   
    m = x.shape[0] 
    cost = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost

    return total_cost


# <a name="toc_40291_2.1"></a>
# ## Resumen del descenso de gradiente
# Hasta ahora en este curso, has desarrollado un modelo lineal que predice $f_{w,b}(x^{(i)})$:
# $$f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{1}$$
# En regresión lineal, usas los datos de entrenamiento para ajustar los parámetros $w$,$b$ minimizando una medida del error entre nuestras predicciones $f_{w,b}(x^{(i)})$ y los datos reales $y^{(i)}$. Esa medida se llama $costo$, $J(w,b)$. Durante el entrenamiento, el costo se mide sobre todas las muestras de entrenamiento $x^{(i)},y^{(i)}$
# $$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2\tag{2}$$ 

# 
# En clase, el *descenso de gradiente* se describió así:
# 
# $$\begin{align*} \text{repetir}&\text{ hasta converger:} \; \lbrace \newline
# \;  w &= w -  \alpha \frac{\partial J(w,b)}{\partial w} \tag{3}  \; \newline 
#  b &= b -  \alpha \frac{\partial J(w,b)}{\partial b}  \newline \rbrace
# \end{align*}$$
# donde los parámetros $w$ y $b$ se actualizan simultáneamente.  
# El gradiente se define como:
# $$
# \begin{align}
# \frac{\partial J(w,b)}{\partial w}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)} \tag{4}\\
#   \frac{\partial J(w,b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) \tag{5}\\
# \end{align}
# $$
# 
# Aquí, *simultáneamente* significa que calculas las derivadas parciales de todos los parámetros antes de actualizar cualquiera de ellos.

# <a name="toc_40291_2.2"></a>
# ## Implementar el descenso de gradiente
# Vas a implementar el algoritmo de descenso de gradiente para una variable. Necesitarás tres funciones: 
# - `compute_gradient`, que implementa las ecuaciones (4) y (5)
# - `compute_cost`, que implementa la ecuación (2) (código del laboratorio anterior)
# - `gradient_descent`, que usa compute_gradient y compute_cost
# 
# Convenciones:
# - Los nombres de las variables de Python que contienen derivadas parciales siguen este patrón: $\frac{\partial J(w,b)}{\partial b}$ será `dj_db`.
# - "w.r.t." significa *With Respect To* ("con respecto a"), como en la derivada parcial de $J(w,b)$ con respecto a $b$.
# 

# <a name="toc_40291_2.3"></a>
# ### compute_gradient
# <a name='ex-01'></a>
# `compute_gradient` implementa las ecuaciones (4) y (5) y devuelve $\frac{\partial J(w,b)}{\partial w}$,$\frac{\partial J(w,b)}{\partial b}$. Los comentarios dentro del código describen las operaciones.

# In[ ]:


def compute_gradient(x, y, w, b): 
    """
    Calcula el gradiente para regresión lineal 
    Args:
      x (ndarray (m,)): Datos, m ejemplos 
      y (ndarray (m,)): valores objetivo
      w,b (scalar)    : parámetros del modelo  
    Returns
      dj_dw (scalar): El gradiente del costo con respecto al parámetro w
      dj_db (scalar): El gradiente del costo con respecto al parámetro b     
     """
    
    # Número de ejemplos de entrenamiento
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    return dj_dw, dj_db


# <br/>

# <img align="left" src="./images/C1_W1_Lab03_lecture_slopes.PNG"   style="width:340px;" > En clase vimos cómo el descenso de gradiente usa la derivada parcial del costo con respecto a un parámetro, evaluada en un punto, para actualizar ese parámetro.   
# Usemos nuestra función `compute_gradient` para calcular y graficar algunas derivadas parciales de la función de costo con respecto a uno de los parámetros, $w$.
# 

# In[ ]:


plt_gradients(x_train,y_train, compute_cost, compute_gradient)
plt.show()


# La gráfica de la izquierda muestra $\frac{\partial J(w,b)}{\partial w}$, es decir, la pendiente de la curva de costo con respecto a $w$ en tres puntos. En el lado derecho de la curva la derivada es positiva, mientras que en el lado izquierdo es negativa. Gracias a la "forma de tazón", las derivadas siempre llevarán al descenso de gradiente hacia el fondo, donde el gradiente es cero.
#  
# En la gráfica de la izquierda, $b$ está fijo en 100. El descenso de gradiente usará tanto $\frac{\partial J(w,b)}{\partial w}$ como $\frac{\partial J(w,b)}{\partial b}$ para actualizar los parámetros. El "gráfico de flechas" (quiver plot) de la derecha permite ver el gradiente de ambos parámetros. El tamaño de las flechas refleja la magnitud del gradiente en cada punto. La dirección y la inclinación de cada flecha reflejan la proporción entre $\frac{\partial J(w,b)}{\partial w}$ y $\frac{\partial J(w,b)}{\partial b}$ en ese punto.
# Observa que el gradiente apunta *en sentido contrario* al mínimo. Revisa la ecuación (3): el gradiente, escalado por alfa, se *resta* del valor actual de $w$ o $b$. Esto mueve el parámetro en la dirección que reduce el costo.

# <a name="toc_40291_2.5"></a>
# ###  Descenso de gradiente
# Ahora que podemos calcular gradientes, el descenso de gradiente descrito en la ecuación (3) se puede implementar en la función `gradient_descent`. Los detalles de la implementación están en los comentarios. Después usarás esta función para encontrar los valores óptimos de $w$ y $b$ con los datos de entrenamiento.

# In[ ]:


def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 
    """
    Ejecuta descenso de gradiente para ajustar w,b. Actualiza w,b dando 
    num_iters pasos de gradiente con tasa de aprendizaje alpha
    
    Args:
      x (ndarray (m,))  : Datos, m ejemplos 
      y (ndarray (m,))  : valores objetivo
      w_in,b_in (scalar): valores iniciales de los parámetros del modelo  
      alpha (float):     Tasa de aprendizaje
      num_iters (int):   número de iteraciones del descenso de gradiente
      cost_function:     función que se llama para calcular el costo
      gradient_function: función que se llama para calcular el gradiente
      
    Returns:
      w (scalar): Valor actualizado del parámetro tras el descenso de gradiente
      b (scalar): Valor actualizado del parámetro tras el descenso de gradiente
      J_history (List): Historial de valores del costo
      p_history (list): Historial de los parámetros [w,b] 
      """
    
    # Listas para guardar el costo J y los parámetros en cada iteración, principalmente para graficar después
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        # Calcular el gradiente con gradient_function
        dj_dw, dj_db = gradient_function(x, y, w , b)     

        # Actualizar los parámetros usando la ecuación (3)
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        # Guardar el costo J en cada iteración
        if i<100000:      # evita agotar los recursos 
            J_history.append( cost_function(x, y, w , b))
            p_history.append([w,b])
        # Imprimir el costo 10 veces a lo largo del proceso (o en cada iteración si son menos de 10)
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteración {i:4}: Costo {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
 
    return w, b, J_history, p_history #devuelve w, b y los historiales de J y [w,b] para graficar


# In[ ]:


# inicializar los parámetros
w_init = 0
b_init = 0
# configuración del descenso de gradiente
iterations = 10000
tmp_alpha = 1.0e-2
# ejecutar el descenso de gradiente
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, 
                                                    iterations, compute_cost, compute_gradient)
print(f"(w,b) encontrados por descenso de gradiente: ({w_final:8.4f},{b_final:8.4f})")


# <img align="left" src="./images/C1_W1_Lab03_lecture_learningrate.PNG"  style="width:340px; padding: 15px; " > 
# Tómate un momento para observar algunas características del proceso de descenso de gradiente que se imprimió arriba.  
# 
# - El costo empieza alto y disminuye rápidamente, como se describió en la diapositiva de la clase.
# - Las derivadas parciales, `dj_dw` y `dj_db`, también se hacen más pequeñas: rápido al principio y luego más lento. Como se mostró en el diagrama de la clase, a medida que el proceso se acerca al "fondo del tazón", el avance es más lento porque la derivada es menor en ese punto.
# - El avance se desacelera aunque la tasa de aprendizaje, alfa, se mantiene fija.

# ### Costo vs. iteraciones del descenso de gradiente 
# Graficar el costo contra las iteraciones es una forma útil de medir el avance del descenso de gradiente. En una ejecución exitosa, el costo siempre debe disminuir. Como el cambio en el costo es muy rápido al principio, conviene graficar el descenso inicial en una escala distinta a la del descenso final. En las gráficas de abajo, fíjate en la escala del costo en los ejes y en el número de iteración.

# In[ ]:


# graficar el costo vs. la iteración  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Costo vs. iteración (inicio)");  ax2.set_title("Costo vs. iteración (final)")
ax1.set_ylabel('Costo')            ;  ax2.set_ylabel('Costo') 
ax1.set_xlabel('Iteración')  ;  ax2.set_xlabel('Iteración') 
plt.show()


# ### Predicciones
# Ahora que encontraste los valores óptimos de los parámetros $w$ y $b$, puedes usar el modelo para predecir el valor de las viviendas con los parámetros aprendidos. Como era de esperar, los valores predichos son prácticamente iguales a los valores de entrenamiento para las mismas casas. Además, la predicción para la casa que no está en los datos de entrenamiento coincide con el valor esperado.

# In[ ]:


print(f"Predicción para una casa de 1000 pies²: {w_final*1.0 + b_final:0.1f} mil dólares")
print(f"Predicción para una casa de 1200 pies²: {w_final*1.2 + b_final:0.1f} mil dólares")
print(f"Predicción para una casa de 2000 pies²: {w_final*2.0 + b_final:0.1f} mil dólares")


# <a name="toc_40291_2.6"></a>
# ## Visualización
# Puedes mostrar el avance del descenso de gradiente durante su ejecución graficando la trayectoria de los parámetros sobre un gráfico de contorno del costo(w,b). 

# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(12, 6))
plt_contour_wgrad(x_train, y_train, p_hist, ax)


# El gráfico de contorno muestra el $costo(w,b)$ en un rango de valores de $w$ y $b$. Los niveles de costo se representan con anillos. Encima, con flechas rojas, está la trayectoria del descenso de gradiente. Algunas cosas a notar:
# - La trayectoria avanza de forma constante (monótona) hacia su objetivo.
# - Los primeros pasos son mucho más grandes que los pasos cerca del objetivo.

# **Haciendo zoom**, podemos ver los pasos finales del descenso de gradiente. Observa que la distancia entre pasos se reduce a medida que el gradiente se acerca a cero.

# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(12, 4))
plt_contour_wgrad(x_train, y_train, p_hist, ax, w_range=[180, 220, 0.5], b_range=[80, 120, 0.5],
            contours=[1,5,10,20],resolution=0.5)


# <a name="toc_40291_2.7.1"></a>
# ### Aumentar la tasa de aprendizaje
# 
# <figure>
#  <img align="left", src="./images/C1_W1_Lab03_alpha_too_big.PNG"   style="width:340px;height:240px;" >
# </figure>
# En clase se discutió cuál es el valor adecuado de la tasa de aprendizaje, $\alpha$, en la ecuación (3). Mientras mayor sea $\alpha$, más rápido convergerá el descenso de gradiente a una solución. Pero si es demasiado grande, el descenso de gradiente divergerá. Arriba tienes un ejemplo de una solución que converge bien.
# 
# Probemos aumentar el valor de $\alpha$ y veamos qué pasa:

# In[ ]:


# inicializar los parámetros
w_init = 0
b_init = 0
# usar un valor grande de alpha
iterations = 10
tmp_alpha = 8.0e-1
# ejecutar el descenso de gradiente
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, 
                                                    iterations, compute_cost, compute_gradient)


# Arriba, $w$ y $b$ rebotan entre valores positivos y negativos, y su valor absoluto aumenta en cada iteración. Además, en cada iteración $\frac{\partial J(w,b)}{\partial w}$ cambia de signo y el costo aumenta en vez de disminuir. Esta es una señal clara de que la *tasa de aprendizaje es demasiado grande* y la solución está divergiendo. 
# Visualicémoslo con una gráfica.

# In[ ]:


plt_divergence(p_hist, J_hist,x_train, y_train)
plt.show()


# La gráfica de la izquierda muestra la evolución de $w$ en los primeros pasos del descenso de gradiente. $w$ oscila entre valores positivos y negativos, y el costo crece rápidamente. El descenso de gradiente actúa sobre $w$ y $b$ simultáneamente, así que se necesita la gráfica 3D de la derecha para ver el panorama completo.

# 
# ## ¡Felicidades!
# En este laboratorio:
# - Profundizaste en los detalles del descenso de gradiente para una variable.
# - Desarrollaste una función para calcular el gradiente.
# - Visualizaste qué es el gradiente.
# - Completaste una función de descenso de gradiente.
# - Usaste el descenso de gradiente para encontrar los parámetros.
# - Examinaste el efecto del tamaño de la tasa de aprendizaje.

# In[ ]:
