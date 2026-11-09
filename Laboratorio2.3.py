#!/usr/bin/env python
# coding: utf-8

# # Laboratorio opcional: Escalado de variables y tasa de aprendizaje (múltiples variables)

# ## Objetivos
# En este laboratorio vas a:
# - Usar las funciones para múltiples variables desarrolladas en el laboratorio anterior
# - Ejecutar descenso de gradiente sobre un conjunto de datos con múltiples variables
# - Explorar el efecto de la *tasa de aprendizaje alfa* en el descenso de gradiente
# - Mejorar el desempeño del descenso de gradiente mediante *escalado de variables* con normalización z-score

# ## Herramientas
# Usarás las funciones desarrolladas en el laboratorio anterior, además de Matplotlib y NumPy. 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import  load_house_data, run_gradient_descent 
from lab_utils_multi import  norm_plot, plt_equal_scale, plot_cost_i_w
from lab_utils_common import dlc
np.set_printoptions(precision=2)
#plt.style.use('./deeplearning.mplstyle')


# ## Notación
# 
# |General <br />  Notación  | Descripción| Python (si aplica) |
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
# | $f_{\mathbf{w},b}(\mathbf{x}^{(i)})$ | Resultado de evaluar el modelo en $\mathbf{x}^{(i)}$, parametrizado por $\mathbf{w},b$: $f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w} \cdot \mathbf{x}^{(i)}+b$  | `f_wb` | 
# |$\frac{\partial J(\mathbf{w},b)}{\partial w_j}$| el gradiente o derivada parcial del costo con respecto al parámetro $w_j$ |`dj_dw[j]`| 
# |$\frac{\partial J(\mathbf{w},b)}{\partial b}$| el gradiente o derivada parcial del costo con respecto al parámetro $b$| `dj_db`|

# #  Planteamiento del problema
# 
# Como en los laboratorios anteriores, usarás el ejemplo de predicción del precio de viviendas. El conjunto de entrenamiento contiene muchos ejemplos con 4 variables (tamaño, habitaciones, pisos y antigüedad), como se muestra en la tabla. Observa que en este laboratorio el tamaño está en pies², mientras que en los primeros laboratorios estaba en miles de pies². Este conjunto de datos es más grande que el del laboratorio anterior.
# 
# Queremos construir un modelo de regresión lineal con estos valores para luego predecir el precio de otras casas; por ejemplo, una casa de 1200 pies², 3 habitaciones, 1 piso y 40 años de antigüedad. 
# 
# ##  Conjunto de datos: 
# | Tamaño (pies²) | Número de habitaciones  | Número de pisos | Antigüedad de la casa | Precio (miles de dólares)  |   
# | ----------------| ------------------- |----------------- |--------------|----------------------- |  
# | 952             | 2                   | 1                | 65           | 271.5                  |  
# | 1244            | 3                   | 2                | 64           | 232                    |  
# | 1947            | 3                   | 2                | 17           | 509.8                  |  
# | ...             | ...                 | ...              | ...          | ...                    |
# 

# In[ ]:


# cargar el conjunto de datos
X_train, y_train = load_house_data()
X_features = ['tamaño (pies²)','habitaciones','pisos','antigüedad']


# Veamos el conjunto de datos y sus variables graficando cada variable contra el precio.

# In[ ]:


fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Precio (miles)")
plt.show()


# Graficar cada variable contra el objetivo, el precio, da una idea de qué variables influyen más en el precio. Arriba se ve que, a mayor tamaño, mayor precio. Las habitaciones y los pisos no parecen tener un gran efecto en el precio. Las casas más nuevas tienen precios más altos que las más viejas.

# <a name="toc_15456_5"></a>
# ## Descenso de gradiente con múltiples variables
# Estas son las ecuaciones que desarrollaste en el laboratorio anterior para el descenso de gradiente con múltiples variables:
# 
# $$\begin{align*} \text{repetir}&\text{ hasta converger:} \; \lbrace \newline\;
# & w_j := w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} \tag{1}  \; & \text{para j = 0..n-1}\newline
# &b\ \ := b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b}  \newline \rbrace
# \end{align*}$$
# 
# donde n es el número de variables, los parámetros $w_j$ y $b$ se actualizan simultáneamente, y donde  
# 
# $$
# \begin{align}
# \frac{\partial J(\mathbf{w},b)}{\partial w_j}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)} \tag{2}  \\
# \frac{\partial J(\mathbf{w},b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) \tag{3}
# \end{align}
# $$
# * m es el número de ejemplos de entrenamiento del conjunto de datos
# 
#     
# * $f_{\mathbf{w},b}(\mathbf{x}^{(i)})$ es la predicción del modelo, mientras que $y^{(i)}$ es el valor objetivo
# 

# ## Tasa de aprendizaje
# <figure>
#     <img src="./images/C1_W2_Lab06_learningrate.PNG" style="width:1200px;" >
# </figure>
# En clase se discutieron algunos aspectos de cómo elegir la tasa de aprendizaje $\alpha$. La tasa de aprendizaje controla el tamaño de la actualización de los parámetros (ver la ecuación (1)) y es la misma para todos los parámetros.  
# 
# Ejecutemos el descenso de gradiente y probemos algunos valores de $\alpha$ con nuestros datos.

# ### $\alpha$ = 9.9e-7

# In[ ]:


#alpha = 9.9e-7
_, _, hist = run_gradient_descent(X_train, y_train, 10, alpha = 9.9e-7)


# Parece que la tasa de aprendizaje es demasiado alta. La solución no converge: el costo *aumenta* en vez de disminuir. Grafiquemos el resultado:

# In[ ]:


plot_cost_i_w(X_train, y_train, hist)


# La gráfica de la derecha muestra el valor de uno de los parámetros, $w_0$. En cada iteración se pasa del valor óptimo y, como resultado, el costo termina *aumentando* en lugar de acercarse al mínimo. Ten en cuenta que esta imagen no es del todo exacta, ya que en cada paso se modifican 4 parámetros y no solo uno. La gráfica muestra solo $w_0$, con los demás parámetros fijos en valores razonables. En esta y en las gráficas siguientes quizá notes que las líneas azul y naranja no coinciden del todo.

# 
# ### $\alpha$ = 9e-7
# Probemos un valor un poco más pequeño y veamos qué pasa.

# In[ ]:


#alpha = 9e-7
_,_,hist = run_gradient_descent(X_train, y_train, 10, alpha = 9e-7)


# El costo disminuye durante toda la ejecución, lo que indica que alfa no es demasiado grande. 

# In[ ]:


plot_cost_i_w(X_train, y_train, hist)


# A la izquierda ves que el costo disminuye, como debe ser. A la derecha ves que $w_0$ todavía oscila alrededor del mínimo, pero el costo baja en cada iteración en lugar de subir. Observa que `dj_dw[0]` cambia de signo en cada iteración, porque `w[0]` salta por encima del valor óptimo.
# Con este valor de alfa, la solución convergerá. Puedes variar el número de iteraciones para ver cómo se comporta.

# ### $\alpha$ = 1e-7
# Probemos un valor aún más pequeño de $\alpha$ y veamos qué pasa.

# In[ ]:


#alpha = 1e-7
_,_,hist = run_gradient_descent(X_train, y_train, 10, alpha = 1e-7)


# El costo disminuye durante toda la ejecución, lo que indica que $\alpha$ no es demasiado grande. 

# In[ ]:


plot_cost_i_w(X_train,y_train,hist)


# A la izquierda ves que el costo disminuye, como debe ser. A la derecha ves que $w_0$ se acerca al mínimo sin oscilar. `dj_w0` es negativo durante toda la ejecución. Esta solución también convergerá.

# ## Escalado de variables 
# <figure>
#     <img src="./images/C1_W2_Lab06_featurescalingheader.PNG" style="width:1200px;" >
# </figure>
# En clase se explicó la importancia de reescalar el conjunto de datos para que las variables tengan rangos similares.
# Si te interesan los detalles de por qué es así, haz clic en el encabezado "Detalles". Si no, la sección siguiente muestra cómo implementar el escalado de variables.

# <details>
# <summary>
#     <font size='3', color='darkgreen'><b>Detalles</b></font>
# </summary>
# 
# Volvamos a la situación con $\alpha$ = 9e-7. Este valor está muy cerca del máximo que podemos usar para $\alpha$ sin que diverja. Esta es una ejecución corta con las primeras iteraciones:
# 
# <figure>
#     <img src="./images/C1_W2_Lab06_ShortRun.PNG" style="width:1200px;" >
# </figure>
# 
# Aunque el costo disminuye, es claro que $w_0$ avanza más rápido que los demás parámetros, porque su gradiente es mucho mayor.
# 
# La siguiente imagen muestra el resultado de una ejecución muy larga con $\alpha$ = 9e-7, que tarda varias horas.
# 
# <figure>
#     <img src="./images/C1_W2_Lab06_LongRun.PNG" style="width:1200px;" >
# </figure>
#     
# Puedes ver que, tras la reducción inicial, el costo disminuye lentamente. Observa la diferencia entre `w0` y `w1`,`w2`,`w3`, y entre `dj_dw0` y `dj_dw1-3`. `w0` llega muy rápido a un valor cercano al final, y `dj_dw0` baja rápidamente a un valor pequeño, lo que indica que `w0` está cerca de su valor final. Los demás parámetros se reducen mucho más lento.
# 
# ¿Por qué pasa esto? ¿Podemos mejorar algo? Mira abajo:
# <figure>
#     <center> <img src="./images/C1_W2_Lab06_scale.PNG"   ></center>
# </figure>   
# 
# La figura muestra por qué los $w$ se actualizan de forma desigual. 
# - $\alpha$ es la misma para todas las actualizaciones de parámetros ($w$ y $b$).
# - El término de error común se multiplica por las variables en el caso de los $w$ (no de $b$).
# - Las variables tienen magnitudes muy distintas, lo que hace que algunas se actualicen mucho más rápido que otras. En este caso, $w_0$ se multiplica por "tamaño (pies²)", que suele ser > 1000, mientras que $w_1$ se multiplica por "número de habitaciones", que suele estar entre 2 y 4. 
#     
# La solución es el escalado de variables.

# En clase se discutieron tres técnicas distintas: 
# - Escalado de variables: básicamente, dividir cada variable positiva entre su valor máximo o, de forma más general, reescalar cada variable usando su mínimo y su máximo con (x-min)/(max-min). La primera forma sirve para variables positivas, es simple y funciona bien para el ejemplo de clase; la segunda funciona para cualquier variable. En ambos casos, los valores quedan entre 0 y 1.
# - Normalización por la media: $x_i := \dfrac{x_i - \mu_i}{max - min} $ 
# - Normalización z-score, que exploraremos a continuación. 

# 
# ### Normalización z-score 
# Después de la normalización z-score, todas las variables tienen media 0 y desviación estándar 1.
# 
# Para implementar la normalización z-score, ajusta los valores de entrada con esta fórmula:
# $$x^{(i)}_j = \dfrac{x^{(i)}_j - \mu_j}{\sigma_j} \tag{4}$$ 
# donde $j$ selecciona una variable, es decir, una columna de la matriz $\mathbf{X}$. $µ_j$ es la media de todos los valores de la variable (j) y $\sigma_j$ es su desviación estándar.
# $$
# \begin{align}
# \mu_j &= \frac{1}{m} \sum_{i=0}^{m-1} x^{(i)}_j \tag{5}\\
# \sigma^2_j &= \frac{1}{m} \sum_{i=0}^{m-1} (x^{(i)}_j - \mu_j)^2  \tag{6}
# \end{align}
# $$
# 
# >**Nota de implementación:** Al normalizar las variables, es importante
# guardar los valores usados en la normalización: la media y la desviación estándar. Después de aprender los parámetros
# del modelo, a menudo queremos predecir el precio de casas que no hemos
# visto antes. Dado un nuevo valor de x (por ejemplo, el área y el número de
# habitaciones), primero debemos normalizarlo con la media y la desviación estándar
# que calculamos a partir del conjunto de entrenamiento.
# 
# **Implementación**

# In[ ]:


def zscore_normalize_features(X):
    """
    calcula X normalizado por columna con z-score
    
    Args:
      X (ndarray (m,n))     : datos de entrada, m ejemplos, n variables
      
    Returns:
      X_norm (ndarray (m,n)): entrada normalizada por columna
      mu (ndarray (n,))     : media de cada variable
      sigma (ndarray (n,))  : desviación estándar de cada variable
    """
    # calcular la media de cada columna/variable
    mu     = np.mean(X, axis=0)                 # mu tendrá forma (n,)
    # calcular la desviación estándar de cada columna/variable
    sigma  = np.std(X, axis=0)                  # sigma tendrá forma (n,)
    # elemento a elemento: restar a cada ejemplo la media de su columna y dividir entre la desviación estándar de esa columna
    X_norm = (X - mu) / sigma      

    return (X_norm, mu, sigma)
 
#verificar nuestro trabajo
#from sklearn.preprocessing import scale
#scale(X_orig, axis=0, with_mean=True, with_std=True, copy=True)


# Veamos los pasos de la normalización z-score. La gráfica de abajo muestra la transformación paso a paso.

# In[ ]:


mu     = np.mean(X_train,axis=0)   
sigma  = np.std(X_train,axis=0) 
X_mean = (X_train - mu)
X_norm = (X_train - mu)/sigma      

fig,ax=plt.subplots(1, 3, figsize=(12, 3))
ax[0].scatter(X_train[:,0], X_train[:,3])
ax[0].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[0].set_title("sin normalizar")
ax[0].axis('equal')

ax[1].scatter(X_mean[:,0], X_mean[:,3])
ax[1].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[1].set_title(r"X - $\mu$")
ax[1].axis('equal')

ax[2].scatter(X_norm[:,0], X_norm[:,3])
ax[2].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[2].set_title(r"normalizado con z-score")
ax[2].axis('equal')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle("distribución de las variables antes, durante y después de normalizar")
plt.show()


# La gráfica muestra la relación entre dos variables del conjunto de entrenamiento, "antigüedad" y "tamaño (pies²)". *Están graficadas con la misma escala*. 
# - Izquierda: sin normalizar. El rango de valores, o la varianza, de "tamaño (pies²)" es mucho mayor que el de la antigüedad.
# - Centro: el primer paso le resta a cada variable su media. Así las variables quedan centradas en cero. Es difícil ver la diferencia en la antigüedad, pero "tamaño (pies²)" ahora está claramente alrededor de cero.
# - Derecha: el segundo paso divide entre la desviación estándar. Así ambas variables quedan centradas en cero y con una escala similar.

# Normalicemos los datos y comparémoslos con los originales.

# In[ ]:


# normalizar las variables originales
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
print(f"Rango (máx - mín) por columna en X original:    {np.ptp(X_train,axis=0)}")   
print(f"Rango (máx - mín) por columna en X normalizado: {np.ptp(X_norm,axis=0)}")


# Con la normalización, el rango (máx - mín) de cada columna pasa de estar en el orden de miles a estar entre 2 y 3.

# In[ ]:


fig,ax=plt.subplots(1, 4, figsize=(12, 3))
for i in range(len(ax)):
    norm_plot(ax[i],X_train[:,i],)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("frecuencia");
fig.suptitle("distribución de las variables antes de normalizar")
plt.show()
fig,ax=plt.subplots(1,4,figsize=(12,3))
for i in range(len(ax)):
    norm_plot(ax[i],X_norm[:,i],)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("frecuencia"); 
fig.suptitle("distribución de las variables después de normalizar")

plt.show()


# Observa que el rango de los datos normalizados (eje x) está centrado en cero y va aproximadamente de -2 a +2. Lo más importante es que el rango es similar para todas las variables.

# Volvamos a ejecutar el descenso de gradiente con los datos normalizados.
# Observa el **valor de alfa, muchísimo más grande**. Esto acelera el descenso de gradiente.

# In[ ]:


w_norm, b_norm, hist = run_gradient_descent(X_norm, y_train, 1000, 1.0e-1, )


# ¡Con las variables escaladas se obtienen resultados muy precisos **muchísimo más rápido**! Observa que, al final de esta ejecución bastante corta, el gradiente de cada parámetro es diminuto. Una tasa de aprendizaje de 0.1 es un buen punto de partida para regresión con variables normalizadas.
# Grafiquemos nuestras predicciones contra los valores objetivo. Ten en cuenta que la predicción se hace con las variables normalizadas, pero la gráfica usa los valores originales.

# In[ ]:


#predecir el objetivo usando las variables normalizadas
m = X_norm.shape[0]
yp = np.zeros(m)
for i in range(m):
    yp[i] = np.dot(X_norm[i], w_norm) + b_norm

    # graficar predicciones y objetivos contra las variables originales    
fig,ax=plt.subplots(1,4,figsize=(12, 3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'objetivo')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],yp,color=dlc["dlorange"], label = 'predicción')
ax[0].set_ylabel("Precio"); ax[0].legend();
fig.suptitle("objetivo vs. predicción con el modelo normalizado con z-score")
plt.show()


# Los resultados se ven bien. Algunos puntos a notar:
# - Con múltiples variables, ya no podemos tener una sola gráfica que muestre los resultados contra las variables.
# - Para generar la gráfica se usaron las variables normalizadas. Cualquier predicción que use parámetros aprendidos de un conjunto de entrenamiento normalizado también debe usar datos normalizados.

# **Predicción**
# El propósito de construir el modelo es usarlo para predecir precios de viviendas que no están en el conjunto de datos. Vamos a predecir el precio de una casa de 1200 pies², 3 habitaciones, 1 piso y 40 años de antigüedad. Recuerda que debes normalizar los datos con la media y la desviación estándar obtenidas al normalizar los datos de entrenamiento. 

# In[ ]:


# Primero, normalizar nuestro ejemplo.
x_house = np.array([1200, 3, 1, 40])
x_house_norm = (x_house - X_mu) / X_sigma
print(x_house_norm)
x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
print(f" precio estimado de una casa de 1200 pies², 3 habitaciones, 1 piso y 40 años de antigüedad = ${x_house_predict*1000:0.0f}")


# **Contornos del costo**  
# <img align="left" src="./images/C1_W2_Lab06_contours.PNG"   style="width:240px;" >Otra forma de ver el escalado de variables es a través de los contornos del costo. Cuando las escalas de las variables no coinciden, el gráfico de contorno del costo en función de los parámetros es asimétrico. 
# 
# En la gráfica de abajo, la escala de los parámetros está igualada. La gráfica de la izquierda es el contorno del costo de w[0] (pies²) contra w[1] (número de habitaciones) antes de normalizar las variables. Es tan asimétrica que no se alcanzan a ver las curvas que cierran los contornos. En cambio, cuando las variables están normalizadas, el contorno del costo es mucho más simétrico. Como resultado, durante el descenso de gradiente cada parámetro puede avanzar al mismo ritmo. 
# 

# In[ ]:


plt_equal_scale(X_train, X_norm, y_train)


# 
# ## ¡Felicidades!
# En este laboratorio:
# - Usaste las funciones de regresión lineal con múltiples variables que desarrollaste en laboratorios anteriores.
# - Exploraste el efecto de la tasa de aprendizaje $\alpha$ en la convergencia.
# - Descubriste el valor del escalado de variables con normalización z-score para acelerar la convergencia.

# ## Agradecimientos
# Los datos de viviendas provienen del [conjunto de datos Ames Housing](http://jse.amstat.org/v19n3/decock.pdf), recopilado por Dean De Cock para su uso en la enseñanza de ciencia de datos.
