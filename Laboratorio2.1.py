#!/usr/bin/env python
# coding: utf-8

# # Laboratorio opcional: Python, NumPy y vectorización
# Una breve introducción a parte de la computación científica que se usa en este curso, en particular a la librería de computación científica NumPy y su uso con Python.
# 
# # Contenido
# - [&nbsp;&nbsp;1.1 Objetivos](#toc_40015_1.1)
# - [&nbsp;&nbsp;1.2 Referencias útiles](#toc_40015_1.2)
# - [2 Python y NumPy <a name='Python and NumPy'></a>](#toc_40015_2)
# - [3 Vectores](#toc_40015_3)
# - [&nbsp;&nbsp;3.1 Conceptos](#toc_40015_3.1)
# - [&nbsp;&nbsp;3.2 Arrays de NumPy](#toc_40015_3.2)
# - [&nbsp;&nbsp;3.3 Creación de vectores](#toc_40015_3.3)
# - [&nbsp;&nbsp;3.4 Operaciones con vectores](#toc_40015_3.4)
# - [4 Matrices](#toc_40015_4)
# - [&nbsp;&nbsp;4.1 Conceptos](#toc_40015_4.1)
# - [&nbsp;&nbsp;4.2 Arrays de NumPy](#toc_40015_4.2)
# - [&nbsp;&nbsp;4.3 Creación de matrices](#toc_40015_4.3)
# - [&nbsp;&nbsp;4.4 Operaciones con matrices](#toc_40015_4.4)
# 

# In[ ]:


import numpy as np    # usar np para numpy es un estándar no oficial
import time


# <a name="toc_40015_1.1"></a>
# ## 1.1 Objetivos
# En este laboratorio vas a:
# - Repasar las funcionalidades de NumPy y Python que se usan en el Curso 1

# <a name="toc_40015_1.2"></a>
# ## 1.2 Referencias útiles
# - Documentación de NumPy, incluida una introducción básica: [NumPy.org](https://NumPy.org/doc/stable/)
# - Un tema más desafiante: [Broadcasting en NumPy](https://NumPy.org/doc/stable/user/basics.broadcasting.html)
# 

# <a name="toc_40015_2"></a>
# # 2 Python y NumPy <a name='Python and NumPy'></a>
# Python es el lenguaje de programación que usaremos en este curso. Tiene un conjunto de tipos de datos numéricos y operaciones aritméticas. NumPy es una librería que amplía las capacidades básicas de Python con un conjunto de datos más rico, que incluye más tipos numéricos, vectores, matrices y muchas funciones para matrices. NumPy y Python funcionan juntos casi sin fricción: los operadores aritméticos de Python funcionan con los tipos de datos de NumPy, y muchas funciones de NumPy aceptan tipos de datos de Python.
# 

# <a name="toc_40015_3"></a>
# # 3 Vectores
# <a name="toc_40015_3.1"></a>
# ## 3.1 Conceptos
# <img align="right" src="./images/C1_W2_Lab04_Vectors.PNG" style="width:340px;" >Los vectores, tal como los usarás en este curso, son arreglos ordenados de números. En la notación, los vectores se escriben con letras minúsculas en negrita, como $\mathbf{x}$. Todos los elementos de un vector son del mismo tipo; un vector no puede, por ejemplo, contener caracteres y números a la vez. Al número de elementos del arreglo se le suele llamar *dimensión*, aunque los matemáticos quizá prefieran *rango*. El vector de la figura tiene dimensión $n$. Los elementos de un vector se pueden referenciar con un índice. En matemáticas, los índices suelen ir de 1 a n; en ciencias de la computación y en estos laboratorios, normalmente van de 0 a n-1. En la notación, cuando se hace referencia a un elemento individual, el índice se indica como subíndice; por ejemplo, el elemento $0$ del vector $\mathbf{x}$ es $x_0$. Observa que en este caso la x no va en negrita.  
# 

# <a name="toc_40015_3.2"></a>
# ## 3.2 Arrays de NumPy
# 
# La estructura de datos básica de NumPy es un *array* n-dimensional e indexable, cuyos elementos son todos del mismo tipo (`dtype`). Notarás enseguida que estamos usando el término "dimensión" con dos significados. Arriba era el número de elementos del vector; aquí, dimensión se refiere al número de índices de un array. Un array unidimensional (1-D) tiene un solo índice. En el Curso 1 representaremos los vectores como arrays 1-D de NumPy. 
# 
#  - Array 1-D, forma (n,): n elementos, indexados de [0] a [n-1]
#  

# <a name="toc_40015_3.3"></a>
# ## 3.3 Creación de vectores
# 

# Las funciones de NumPy para crear datos suelen tener como primer parámetro la forma (shape) del objeto. Puede ser un solo valor, para un resultado 1-D, o una tupla (n,m,...) que especifica la forma del resultado. Abajo hay ejemplos de cómo crear vectores con estas funciones.

# In[ ]:


# Funciones de NumPy que reservan memoria y llenan los arrays con valores
a = np.zeros(4);                print(f"np.zeros(4) :   a = {a}, forma de a = {a.shape}, tipo de dato de a = {a.dtype}")
a = np.zeros((4,));             print(f"np.zeros(4,) :  a = {a}, forma de a = {a.shape}, tipo de dato de a = {a.dtype}")
a = np.random.random_sample(4); print(f"np.random.random_sample(4): a = {a}, forma de a = {a.shape}, tipo de dato de a = {a.dtype}")


# Algunas funciones de creación de datos no reciben una tupla de forma:

# In[ ]:


# Funciones de NumPy que reservan memoria y llenan los arrays con valores, pero no aceptan la forma como argumento
a = np.arange(4.);              print(f"np.arange(4.):     a = {a}, forma de a = {a.shape}, tipo de dato de a = {a.dtype}")
a = np.random.rand(4);          print(f"np.random.rand(4): a = {a}, forma de a = {a.shape}, tipo de dato de a = {a.dtype}")


# También se pueden especificar los valores manualmente. 

# In[ ]:


# Funciones de NumPy que reservan memoria y llenan con valores especificados por el usuario
a = np.array([5,4,3,2]);  print(f"np.array([5,4,3,2]):  a = {a},     forma de a = {a.shape}, tipo de dato de a = {a.dtype}")
a = np.array([5.,4,3,2]); print(f"np.array([5.,4,3,2]): a = {a}, forma de a = {a.shape}, tipo de dato de a = {a.dtype}")


# Todos estos ejemplos crearon un vector unidimensional `a` con cuatro elementos. `a.shape` devuelve las dimensiones. Aquí vemos que a.shape = `(4,)`, lo que indica un array 1-D con 4 elementos.  

# <a name="toc_40015_3.4"></a>
# ## 3.4 Operaciones con vectores
# Exploremos algunas operaciones con vectores.
# <a name="toc_40015_3.4.1"></a>
# ### 3.4.1 Indexación
# Se puede acceder a los elementos de un vector mediante indexación y rebanado (slicing). NumPy ofrece un conjunto muy completo de herramientas de indexación y rebanado; aquí exploraremos solo lo básico que se necesita para el curso. Consulta [Slicing and Indexing](https://NumPy.org/doc/stable/reference/arrays.indexing.html) para más detalles.  
# **Indexar** significa referirse a *un elemento* de un array por su posición dentro del array.  
# **Rebanar** (slicing) significa obtener un *subconjunto* de elementos de un array según sus índices.  
# NumPy empieza a indexar en cero, así que el 3er elemento de un vector $\mathbf{a}$ es `a[2]`.

# In[ ]:


#operaciones de indexación en vectores 1-D
a = np.arange(10)
print(a)

#acceder a un elemento
print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Acceder a un elemento devuelve un escalar")

# acceder al último elemento; los índices negativos cuentan desde el final
print(f"a[-1] = {a[-1]}")

#los índices deben estar dentro del rango del vector; si no, producen un error
try:
    c = a[10]
except Exception as e:
    print("El mensaje de error que verás es:")
    print(e)


# <a name="toc_40015_3.4.2"></a>
# ### 3.4.2 Rebanado (slicing)
# El rebanado crea un array de índices usando un conjunto de tres valores (`inicio:fin:paso`). También es válido usar solo algunos de ellos. Se entiende mejor con ejemplos:

# In[ ]:


#operaciones de rebanado en vectores
a = np.arange(10)
print(f"a         = {a}")

#acceder a 5 elementos consecutivos (inicio:fin:paso)
c = a[2:7:1];     print("a[2:7:1] = ", c)

# acceder a 3 elementos separados de dos en dos 
c = a[2:7:2];     print("a[2:7:2] = ", c)

# acceder a todos los elementos desde el índice 3 en adelante
c = a[3:];        print("a[3:]    = ", c)

# acceder a todos los elementos antes del índice 3
c = a[:3];        print("a[:3]    = ", c)

# acceder a todos los elementos
c = a[:];         print("a[:]     = ", c)


# <a name="toc_40015_3.4.3"></a>
# ### 3.4.3 Operaciones con un solo vector
# Hay varias operaciones útiles que se aplican a un solo vector.

# In[ ]:


a = np.array([1,2,3,4])
print(f"a             : {a}")
# negar los elementos de a
b = -a 
print(f"b = -a        : {b}")

# sumar todos los elementos de a; devuelve un escalar
b = np.sum(a) 
print(f"b = np.sum(a) : {b}")

b = np.mean(a)
print(f"b = np.mean(a): {b}")

b = a**2
print(f"b = a**2      : {b}")


# <a name="toc_40015_3.4.4"></a>
# ### 3.4.4 Operaciones elemento a elemento entre vectores
# La mayoría de las operaciones aritméticas, lógicas y de comparación de NumPy también se aplican a vectores. Estos operadores funcionan elemento por elemento. Por ejemplo: 
# $$ c_i = a_i + b_i $$

# In[ ]:


a = np.array([ 1, 2, 3, 4])
b = np.array([-1,-2, 3, 4])
print(f"Los operadores binarios funcionan elemento a elemento: {a + b}")


# Por supuesto, para que esto funcione correctamente, los vectores deben tener el mismo tamaño:

# In[ ]:


#intentar una operación con vectores de distinto tamaño
c = np.array([1, 2])
try:
    d = a + c
except Exception as e:
    print("El mensaje de error que verás es:")
    print(e)


# <a name="toc_40015_3.4.5"></a>
# ### 3.4.5 Operaciones entre un escalar y un vector
# Los vectores se pueden "escalar" con valores escalares. Un valor escalar es simplemente un número. El escalar multiplica todos los elementos del vector.

# In[ ]:


a = np.array([1, 2, 3, 4])

# multiplicar a por un escalar
b = 5 * a 
print(f"b = 5 * a : {b}")


# <a name="toc_40015_3.4.6"></a>
# ### 3.4.6 Producto punto entre vectores
# El producto punto es un pilar del álgebra lineal y de NumPy. Es una operación que se usa mucho en este curso y conviene entenderla bien. El producto punto se muestra abajo.

# <img src="./images/C1_W2_Lab04_dot_notrans.gif" width=800> 

# El producto punto multiplica los valores de dos vectores elemento a elemento y luego suma los resultados.
# El producto punto requiere que los dos vectores tengan la misma dimensión. 

# Implementemos nuestra propia versión del producto punto:
# 
# **Usando un ciclo for**, implementa una función que devuelva el producto punto de dos vectores. Dadas las entradas $a$ y $b$, la función debe devolver:
# $$ x = \sum_{i=0}^{n-1} a_i b_i $$
# Supón que `a` y `b` tienen la misma forma.

# In[ ]:


def my_dot(a, b): 
    """
    Calcula el producto punto de dos vectores
 
    Args:
      a (ndarray (n,)):  vector de entrada 
      b (ndarray (n,)):  vector de entrada con la misma dimensión que a
    
    Returns:
      x (scalar): 
    """
    x=0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x


# In[ ]:


# prueba 1-D
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
print(f"my_dot(a, b) = {my_dot(a, b)}")


# Observa que se espera que el producto punto devuelva un valor escalar. 
# 
# Probemos las mismas operaciones usando `np.dot`.  

# In[ ]:


# prueba 1-D
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
c = np.dot(a, b)
print(f"NumPy 1-D np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape} ") 
c = np.dot(b, a)
print(f"NumPy 1-D np.dot(b, a) = {c}, np.dot(b, a).shape = {c.shape} ")


# Verás que los resultados en 1-D coinciden con nuestra implementación.

# <a name="toc_40015_3.4.7"></a>
# ### 3.4.7 La necesidad de velocidad: vectorización vs. ciclo for
# Usamos la librería NumPy porque mejora la velocidad y la eficiencia en el uso de memoria. Demostrémoslo:

# In[ ]:


np.random.seed(1)
a = np.random.rand(10000000)  # arrays muy grandes
b = np.random.rand(10000000)

tic = time.time()  # registrar el tiempo de inicio
c = np.dot(a, b)
toc = time.time()  # registrar el tiempo de fin

print(f"np.dot(a, b) =  {c:.4f}")
print(f"Duración de la versión vectorizada: {1000*(toc-tic):.4f} ms ")

tic = time.time()  # registrar el tiempo de inicio
c = my_dot(a,b)
toc = time.time()  # registrar el tiempo de fin

print(f"my_dot(a, b) =  {c:.4f}")
print(f"Duración de la versión con ciclo: {1000*(toc-tic):.4f} ms ")

del(a);del(b)  #eliminar estos arrays grandes de la memoria


# Como ves, la vectorización acelera mucho el cálculo en este ejemplo. Esto se debe a que NumPy aprovecha mejor el paralelismo de datos disponible en el hardware. Las GPU y las CPU modernas implementan canales SIMD (Single Instruction, Multiple Data: una instrucción, múltiples datos), que permiten ejecutar varias operaciones en paralelo. Esto es fundamental en Machine Learning, donde los conjuntos de datos suelen ser muy grandes.

# <a name="toc_12345_3.4.8"></a>
# ### 3.4.8 Operaciones entre vectores en el Curso 1
# Las operaciones entre vectores aparecerán con frecuencia en el Curso 1. Esta es la razón:
# - De aquí en adelante, nuestros ejemplos se guardarán en un array, `X_train`, de dimensión (m,n). Esto se explicará mejor en contexto, pero por ahora es importante notar que es un array bidimensional, es decir, una matriz (ver la siguiente sección sobre matrices).
# - `w` será un vector unidimensional de forma (n,).
# - Haremos operaciones recorriendo los ejemplos y extrayendo cada uno para trabajarlo individualmente, indexando X. Por ejemplo: `X[i]`
# - `X[i]` devuelve un valor de forma (n,), un vector unidimensional. Por lo tanto, las operaciones con `X[i]` suelen ser operaciones entre vectores.  
# 
# Es una explicación algo larga, pero al trabajar con vectores es importante entender y hacer coincidir las formas de los operandos.

# In[ ]:


# mostrar un ejemplo típico del Curso 1
X = np.array([[1],[2],[3],[4]])
w = np.array([2])
c = np.dot(X[1], w)

print(f"X[1] tiene forma {X[1].shape}")
print(f"w tiene forma {w.shape}")
print(f"c tiene forma {c.shape}")


# <a name="toc_40015_4"></a>
# # 4 Matrices
# 

# <a name="toc_40015_4.1"></a>
# ## 4.1 Conceptos
# Las matrices son arrays bidimensionales. Todos los elementos de una matriz son del mismo tipo. En la notación, las matrices se escriben con letras mayúsculas en negrita, como $\mathbf{X}$. En este y otros laboratorios, `m` suele ser el número de filas y `n` el número de columnas. Los elementos de una matriz se referencian con un índice bidimensional. En matemáticas, los índices suelen ir de 1 a n; en ciencias de la computación y en estos laboratorios, van de 0 a n-1.  
# <figure>
#     <center> <img src="./images/C1_W2_Lab04_Matrices.PNG"  alt='missing'  width=900><center/>
#     <figcaption> Notación general de matrices: el 1er índice es la fila y el 2do es la columna </figcaption>
# <figure/>

# <a name="toc_40015_4.2"></a>
# ## 4.2 Arrays de NumPy
# 
# La estructura de datos básica de NumPy es un *array* n-dimensional e indexable, cuyos elementos son todos del mismo tipo (`dtype`), como se describió antes. Las matrices tienen un índice bidimensional (2-D) [m,n].
# 
# En el Curso 1, las matrices 2-D se usan para guardar los datos de entrenamiento. Los datos de entrenamiento tienen $m$ ejemplos por $n$ variables, lo que forma un array (m,n). El Curso 1 no hace operaciones directamente sobre matrices; normalmente extrae un ejemplo como vector y opera sobre él. A continuación repasarás: 
# - creación de datos
# - rebanado e indexación

# <a name="toc_40015_4.3"></a>
# ## 4.3 Creación de matrices
# Las mismas funciones que crean vectores 1-D crean arrays 2-D o n-D. Aquí hay algunos ejemplos.
# 

# Abajo se pasa la tupla de forma para obtener un resultado 2-D. Observa cómo NumPy usa corchetes para indicar cada dimensión. Observa también que, al imprimir, NumPy muestra una fila por línea.
# 

# In[ ]:


a = np.zeros((1, 5))                                       
print(f"forma de a = {a.shape}, a = {a}")                     

a = np.zeros((2, 1))                                                                   
print(f"forma de a = {a.shape}, a = {a}") 

a = np.random.random_sample((1, 1))  
print(f"forma de a = {a.shape}, a = {a}") 


# También se pueden especificar los datos manualmente. Las dimensiones se indican con corchetes adicionales, siguiendo el mismo formato que se ve al imprimir.

# In[ ]:


# Funciones de NumPy que reservan memoria y llenan con valores especificados por el usuario
a = np.array([[5], [4], [3]]);   print(f" forma de a = {a.shape}, np.array: a = {a}")
a = np.array([[5],   # También se pueden
              [4],   # separar los valores
              [3]]); # en filas distintas
print(f" forma de a = {a.shape}, np.array: a = {a}")


# <a name="toc_40015_4.4"></a>
# ## 4.4 Operaciones con matrices
# Exploremos algunas operaciones con matrices.

# <a name="toc_40015_4.4.1"></a>
# ### 4.4.1 Indexación
# 

# Las matrices tienen un segundo índice. Los dos índices describen [fila, columna]. Al acceder, se puede obtener un elemento o una fila/columna. Mira abajo:

# In[ ]:


#operaciones de indexación en matrices
a = np.arange(6).reshape(-1, 2)   #reshape es una forma práctica de crear matrices
print(f"a.shape: {a.shape}, \na= {a}")

#acceder a un elemento
print(f"\na[2,0].shape:   {a[2, 0].shape}, a[2,0] = {a[2, 0]},     type(a[2,0]) = {type(a[2, 0])} Acceder a un elemento devuelve un escalar\n")

#acceder a una fila
print(f"a[2].shape:   {a[2].shape}, a[2]   = {a[2]}, type(a[2])   = {type(a[2])}")


# Vale la pena prestar atención al último ejemplo: acceder a una matriz indicando solo la fila devuelve un *vector 1-D*.

# **Reshape**  
# El ejemplo anterior usó [reshape](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html) para darle forma al array.  
# `a = np.arange(6).reshape(-1, 2) `   
# Esta línea primero crea un *vector 1-D* de seis elementos y luego lo transforma en un array *2-D* con el comando reshape. También se podría haber escrito:  
# `a = np.arange(6).reshape(3, 2) `  
# para obtener el mismo array de 3 filas y 2 columnas.
# El argumento -1 le indica a la función que calcule el número de filas a partir del tamaño del array y del número de columnas.
# 

# <a name="toc_40015_4.4.2"></a>
# ### 4.4.2 Rebanado (slicing)
# El rebanado crea un array de índices usando un conjunto de tres valores (`inicio:fin:paso`). También es válido usar solo algunos de ellos. Se entiende mejor con ejemplos:

# In[ ]:


#operaciones de rebanado en arrays 2-D
a = np.arange(20).reshape(-1, 10)
print(f"a = \n{a}")

#acceder a 5 elementos consecutivos (inicio:fin:paso)
print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "un array 1-D")

#acceder a 5 elementos consecutivos (inicio:fin:paso) en las dos filas
print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "un array 2-D")

# acceder a todos los elementos
print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)

# acceder a todos los elementos de una fila (uso muy común)
print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "un array 1-D")
# equivale a
print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "un array 1-D")


# <a name="toc_40015_5.0"></a>
# ## ¡Felicidades!
# En este laboratorio dominaste las funcionalidades de Python y NumPy que se necesitan para el Curso 1.

# In[ ]:
