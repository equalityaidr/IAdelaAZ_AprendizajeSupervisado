#!/usr/bin/env python
# coding: utf-8


# 
# <figure>
#  <img src="./images/C1_W1_L3_S1_Lecture_b.png"   style="width:600px;height:200px;">
# </figure>

# ## Objetivos
# En este laboratorio vas a:
# - Aprender a implementar el modelo $f_{w,b}$ para regresión lineal con una variable

# ## Notación
# Aquí tienes un resumen de la notación que encontrarás en este ejercicio
# 
# |General <img width=70/> <br />  Notación  <img width=70/> | Descripción<img width=350/>| Python (si aplica) |
# |: ------------|: ------------------------------------------------------------||
# | $a$ | escalar, sin negrita                                                      ||
# | $\mathbf{a}$ | vector, en negrita                                                      ||
# | **Regresión** |         |    |     |
# |  $\mathbf{x}$ | Valores de la variable en los datos de entrenamiento (en este lab: Tamaño (1000 pies²))  | `x_train` |   
# |  $\mathbf{y}$  | Valores objetivo de los ejemplos de entrenamiento (en este lab: Precio (miles de dólares))  | `y_train` 
# |  $x^{(i)}$, $y^{(i)}$ | $i$-ésimo ejemplo de entrenamiento | `x_i`, `y_i`|
# | m | Número de ejemplos de entrenamiento | `m`|
# |  $w$  |  parámetro: peso                                 | `w`    |
# |  $b$           |  parámetro: sesgo                                           | `b`    |     
# | $f_{w,b}(x^{(i)})$ | Resultado de evaluar el modelo en $x^{(i)}$, parametrizado por $w,b$: $f_{w,b}(x^{(i)}) = wx^{(i)}+b$  | `f_wb` | 
# 

# ## Herramientas
# En este laboratorio usarás: 
# - NumPy, una librería de computación científica
# - Matplotlib, una librería de visualización de datos

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('./deeplearning.mplstyle')


# # Planteamiento del problema
# <img align="left" src="./images/C1_W1_L3_S1_trainingdata.png"    style=" width:380px; padding: 10px;  " /> 
# 
# Como en la clase, usarás el ejemplo de predicción del precio de viviendas.  
# Este laboratorio usa un conjunto de datos sencillo con solo dos puntos: una casa de 1000 pies cuadrados (pies²) vendida en \\$300,000 y una casa de 2000 pies cuadrados vendida en \\$500,000. Estos dos puntos constituyen nuestros *datos o conjunto de entrenamiento*. En este laboratorio, las unidades de tamaño son 1000 pies² y las unidades de precio son miles de dólares.
# 
# | Tamaño (1000 pies²)     | Precio (miles de dólares) |
# | -------------------| ------------------------ |
# | 1.0               | 300                      |
# | 2.0               | 500                      |
# 
# Quieres ajustar un modelo de regresión lineal (la línea recta azul de la figura) a través de estos dos puntos, para luego poder predecir el precio de otras casas; por ejemplo, una casa de 1200 pies².
# 

# Ejecuta la siguiente celda de código para crear tus variables `x_train` y `y_train`. Los datos se almacenan en arreglos (arrays) unidimensionales de NumPy.

# In[ ]:


# x_train es la variable de entrada (tamaño en miles de pies cuadrados)
# y_train es el objetivo (precio en miles de dólares)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")


# >**Nota**: El curso usará con frecuencia el formato de salida 'f-string' de Python, descrito [aquí](https://docs.python.org/3/tutorial/inputoutput.html), al imprimir resultados. El contenido entre llaves se evalúa al generar la salida.

# ### Número de ejemplos de entrenamiento `m`
# Usarás `m` para denotar el número de ejemplos de entrenamiento. Los arrays de NumPy tienen un atributo `.shape`. `x_train.shape` devuelve una tupla de Python con una entrada por cada dimensión. `x_train.shape[0]` es la longitud del array, es decir, el número de ejemplos, como se muestra abajo.

# In[ ]:


# m es el número de ejemplos de entrenamiento
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"El número de ejemplos de entrenamiento es: {m}")


# También se puede usar la función `len()` de Python, como se muestra abajo.

# In[ ]:


# m es el número de ejemplos de entrenamiento
m = len(x_train)
print(f"El número de ejemplos de entrenamiento es: {m}")


# ### Ejemplo de entrenamiento `x_i, y_i`
# 
# Usarás (x$^{(i)}$, y$^{(i)}$) para denotar el $i$-ésimo ejemplo de entrenamiento. Como Python indexa desde cero, (x$^{(0)}$, y$^{(0)}$) es (1.0, 300.0) y (x$^{(1)}$, y$^{(1)}$) es (2.0, 500.0). 
# 
# Para acceder a un valor de un array de NumPy, se indexa el array con la posición deseada. Por ejemplo, la sintaxis para acceder a la posición cero de `x_train` es `x_train[0]`.
# Ejecuta el siguiente bloque de código para obtener el $i$-ésimo ejemplo de entrenamiento.

# In[ ]:


i = 0 # Cambia esto a 1 para ver (x^1, y^1)

x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")


# ### Graficar los datos

# Puedes graficar estos dos puntos con la función `scatter()` de la librería `matplotlib`, como se muestra en la celda de abajo. 
# - Los argumentos `marker` y `c` muestran los puntos como cruces rojas (por defecto son puntos azules).
# 
# Puedes usar otras funciones de `matplotlib` para definir el título y las etiquetas que se mostrarán.

# In[ ]:


# Graficar los puntos de datos
plt.scatter(x_train, y_train, marker='x', c='r')
# Definir el título
plt.title("Precios de viviendas")
# Definir la etiqueta del eje y
plt.ylabel('Precio (miles de dólares)')
# Definir la etiqueta del eje x
plt.xlabel('Tamaño (1000 pies²)')
plt.show()


# ## Función del modelo
# 
# <img align="left" src="./images/C1_W1_L3_S1_model.png"     style=" width:380px; padding: 10px; " > Como se explicó en clase, la función del modelo de regresión lineal (una función que mapea de `x` a `y`) se representa como 
# 
# $$ f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{1}$$
# 
# La fórmula anterior es la forma de representar líneas rectas: distintos valores de $w$ y $b$ producen distintas líneas rectas en la gráfica. <br/> <br/> <br/> <br/> <br/> 
# 
# Construyamos una mejor intuición con los bloques de código siguientes. Empecemos con $w = 100$ y $b = 100$. 
# 
# **Nota: Puedes volver a esta celda para ajustar los parámetros w y b del modelo**

# In[ ]:


w = 100
b = 100
print(f"w: {w}")
print(f"b: {b}")


# Ahora calculemos el valor de $f_{w,b}(x^{(i)})$ para tus dos puntos de datos. Puedes escribirlo explícitamente para cada punto así: 
# 
# para $x^{(0)}$, `f_wb = w * x[0] + b`
# 
# para $x^{(1)}$, `f_wb = w * x[1] + b`
# 
# Con muchos puntos de datos, esto se vuelve engorroso y repetitivo. Por eso, puedes calcular la salida de la función en un ciclo `for`, como en la función `compute_model_output` de abajo.
# > **Nota**: La descripción del argumento `(ndarray (m,))` indica un array n-dimensional de NumPy con forma (m,). `(scalar)` indica un argumento sin dimensiones, solo una magnitud.  
# > **Nota**: `np.zeros(n)` devuelve un array unidimensional de NumPy con $n$ entradas   
# 

# In[ ]:


def compute_model_output(x, w, b):
    """
    Calcula la predicción de un modelo lineal
    Args:
      x (ndarray (m,)): Datos, m ejemplos 
      w,b (scalar)    : parámetros del modelo  
    Returns
      f_wb (ndarray (m,)): predicción del modelo
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb


# Ahora llamemos a la función `compute_model_output` y grafiquemos el resultado.

# In[ ]:


tmp_f_wb = compute_model_output(x_train, w, b,)

# Graficar la predicción del modelo
plt.plot(x_train, tmp_f_wb, c='b',label='Nuestra predicción')

# Graficar los puntos de datos
plt.scatter(x_train, y_train, marker='x', c='r',label='Valores reales')

# Definir el título
plt.title("Precios de viviendas")
# Definir la etiqueta del eje y
plt.ylabel('Precio (miles de dólares)')
# Definir la etiqueta del eje x
plt.xlabel('Tamaño (1000 pies²)')
plt.legend()
plt.show()


# Como puedes ver, usar $w = 100$ y $b = 100$ *no* produce una línea que se ajuste a nuestros datos. 
# 
# ### Reto
# Experimenta con distintos valores de $w$ y $b$. ¿Qué valores deberían tener para obtener una línea que se ajuste a los datos?
# 
# #### Consejo:
# Haz clic en "Pistas" (en verde) para ver algunas ideas para elegir b y w.

# <details>
# <summary>
#     <font size='3', color='darkgreen'><b>Pistas</b></font>
# </summary>
#     <p>
#     <ul>
#         <li>Prueba con $w = 200$ y $b = 100$ </li>
#     </ul>
#     </p>

# ### Predicción
# Ahora que tenemos un modelo, podemos usarlo para hacer la predicción que buscábamos. Vamos a predecir el precio de una casa de 1200 pies². Como las unidades de $x$ son miles de pies², $x$ es 1.2.
# 

# In[ ]:


w = 200                         
b = 100    
x_i = 1.2
cost_1200sqft = w * x_i + b    

print(f"${cost_1200sqft:.0f} mil dólares")


# # ¡Felicidades!
# En este laboratorio aprendiste que:
#  - La regresión lineal construye un modelo que establece una relación entre variables de entrada (features) y objetivos (targets)
#      - En el ejemplo anterior, la variable de entrada era el tamaño de la casa y el objetivo era su precio
#      - En la regresión lineal simple, el modelo tiene dos parámetros, $w$ y $b$, cuyos valores se "ajustan" usando *datos de entrenamiento*.
#      - Una vez determinados los parámetros del modelo, este puede usarse para hacer predicciones sobre datos nuevos.

# In[ ]:
