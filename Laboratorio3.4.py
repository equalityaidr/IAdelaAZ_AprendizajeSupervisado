#!/usr/bin/env python
# coding: utf-8

# # Laboratorio opcional: Regresión logística, pérdida logística
# 
# En este laboratorio sin calificación vas a:
# - Explorar por qué la pérdida de error cuadrático no es adecuada para la regresión logística
# - Explorar la función de pérdida logística

# In[ ]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'widget')
import matplotlib.pyplot as plt
from plt_logistic_loss import  plt_logistic_cost, plt_two_logistic_loss_curves, plt_simple_example
from plt_logistic_loss import soup_bowl, plt_logistic_squared_error
plt.style.use('./deeplearning.mplstyle')


# ## ¿Error cuadrático para la regresión logística?
# <img align="left" src="./images/C1_W3_SqErrorVsLogistic.png"     style=" width:400px; padding: 10px; " > Recuerda que en regresión **lineal** usamos la **función de costo de error cuadrático**.
# La ecuación del costo de error cuadrático con una variable es:
#   $$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \tag{1}$$ 
#  
# donde 
#   $$f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{2}$$
# 

# Recuerda que el costo de error cuadrático tenía una propiedad muy útil: seguir la derivada del costo lleva al mínimo.

# In[ ]:


soup_bowl()


# Esta función de costo funcionó bien para la regresión lineal, así que es natural considerarla también para la regresión logística. Sin embargo, como señala la diapositiva de arriba, ahora $f_{wb}(x)$ tiene un componente no lineal, la función sigmoide: $f_{w,b}(x^{(i)}) = sigmoid(wx^{(i)} + b )$. Probemos el costo de error cuadrático con el ejemplo de un laboratorio anterior, ahora incluyendo la sigmoide.
# 
# Estos son nuestros datos de entrenamiento:

# In[ ]:


x_train = np.array([0., 1, 2, 3, 4, 5],dtype=np.longdouble)
y_train = np.array([0,  0, 0, 1, 1, 1],dtype=np.longdouble)
plt_simple_example(x_train, y_train)


# Ahora obtengamos una gráfica de superficie del costo usando el *costo de error cuadrático*:
#   $$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 $$ 
#  
# donde 
#   $$f_{w,b}(x^{(i)}) = sigmoid(wx^{(i)} + b )$$
# 

# In[ ]:


plt.close('all')
plt_logistic_squared_error(x_train,y_train)
plt.show()


# Aunque produce una gráfica bastante interesante, ¡la superficie no es ni de lejos tan suave como el "tazón de sopa" de la regresión lineal!    
# 
# La regresión logística requiere una función de costo más adecuada a su naturaleza no lineal. Esto empieza con una función de pérdida, que se describe a continuación.

# ## Función de pérdida logística
# <img align="left" src="./images/C1_W3_LogisticLoss_a.png"     style=" width:250px; padding: 2px; " >
# <img align="left" src="./images/C1_W3_LogisticLoss_b.png"     style=" width:250px; padding: 2px; " >
# <img align="left" src="./images/C1_W3_LogisticLoss_c.png"     style=" width:250px; padding: 2px; " > 

# La regresión logística usa una función de pérdida más adecuada para tareas de clasificación, donde el objetivo es 0 o 1 y no cualquier número. 
# 
# >**Nota sobre las definiciones:** En este curso se usan estas definiciones:  
# La **pérdida** mide la diferencia entre la predicción para un solo ejemplo y su valor objetivo, mientras que el  
# **costo** mide las pérdidas sobre todo el conjunto de entrenamiento.
# 
# 
# Se define así: 
# * $loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)})$ es la pérdida para un solo punto de datos, que es:
# 
# \begin{equation}
#   loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = \begin{cases}
#     - \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) & \text{si $y^{(i)}=1$}\\
#     - \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) & \text{si $y^{(i)}=0$}
#   \end{cases}
# \end{equation}
# 
# 
# * $f_{\mathbf{w},b}(\mathbf{x}^{(i)})$ es la predicción del modelo, mientras que $y^{(i)}$ es el valor objetivo.
# 
# * $f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = g(\mathbf{w} \cdot\mathbf{x}^{(i)}+b)$, donde $g$ es la función sigmoide.
# * Convención de notación: `log` significa logaritmo natural. 
# 
# La característica distintiva de esta función de pérdida es que usa dos curvas separadas: una para cuando el objetivo es cero ($y=0$) y otra para cuando el objetivo es uno ($y=1$). Juntas, estas curvas dan el comportamiento que se espera de una función de pérdida: vale cero cuando la predicción coincide con el objetivo y aumenta rápidamente a medida que la predicción se aleja del objetivo. Observa las curvas de abajo:

# In[ ]:


plt_two_logistic_loss_curves()


# Juntas, las curvas se parecen a la curva cuadrática de la pérdida de error cuadrático. Observa que el eje x es $f_{\mathbf{w},b}$, la salida de una sigmoide, que siempre está estrictamente entre 0 y 1.

# La función de pérdida anterior se puede reescribir para que sea más fácil de implementar.
#     $$loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = (-y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)$$
#   
# Es una ecuación de aspecto bastante intimidante. Se vuelve más sencilla si consideras que $y^{(i)}$ solo puede tomar dos valores, 0 y 1. Así, la ecuación se puede ver en dos partes:  
# cuando $ y^{(i)} = 0$, se elimina el término de la izquierda:
# $$
# \begin{align}
# loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), 0) &= (-(0) \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - 0\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \\
# &= -\log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)
# \end{align}
# $$
# y cuando $ y^{(i)} = 1$, se elimina el término de la derecha:
# $$
# \begin{align}
#   loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), 1) &=  (-(1) \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - 1\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)\\
#   &=  -\log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)
# \end{align}
# $$
# 
# Con esta nueva función de pérdida logística se puede construir una función de costo que incorpore la pérdida de todos los ejemplos. Ese será el tema del próximo laboratorio. Por ahora, veamos la curva del costo en función de los parámetros para el ejemplo sencillo que usamos arriba:

# In[ ]:


plt.close('all')
cst = plt_logistic_cost(x_train,y_train)


# ¡Esta curva es muy adecuada para el descenso de gradiente! No tiene mesetas, mínimos locales ni discontinuidades. Observa que no es un tazón como en el caso del error cuadrático. Se grafican tanto el costo como el logaritmo del costo para mostrar que, cuando el costo es pequeño, la curva sigue teniendo pendiente y sigue bajando. Recuerda: puedes rotar estas gráficas con el mouse.

# ## ¡Felicidades!  
# En este laboratorio:
#  - Comprobaste que la función de pérdida de error cuadrático no es adecuada para tareas de clasificación.
#  - Desarrollaste y examinaste la función de pérdida logística, que **sí** es adecuada para tareas de clasificación.
# 
#
