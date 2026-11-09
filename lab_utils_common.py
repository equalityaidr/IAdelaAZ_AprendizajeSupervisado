"""
lab_utils_common
   contiene funciones y definiciones de variables comunes
   que usan todos los laboratorios de esta semana.
   en cambio, las funciones de graficación grandes y específicas están en archivos
   separados y, por lo general, se importan en la semana en que se usan.
   esos archivos importan este archivo
"""
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from ipywidgets import Output

np.set_printoptions(precision=2)

dlc = dict(dlblue = '#0096ff', dlorange = '#FF9300', dldarkred='#C00000', dlmagenta='#FF40FF', dlpurple='#7030A0')
dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0'
dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]
plt.style.use('./deeplearning.mplstyle')

def sigmoid(z):
    """
    Calcula la sigmoide de z

    Parámetros
    ----------
    z : array_like
        Un escalar o un array de numpy de cualquier tamaño.

    Devuelve
    -------
     g : array_like
         sigmoid(z)
    """
    z = np.clip( z, -500, 500 )           # protege contra desbordamiento (overflow)
    g = 1.0/(1.0+np.exp(-z))

    return g

##########################################################
# Funciones de regresión
##########################################################

def predict_logistic(X, w, b):
    """ hace la predicción """
    return sigmoid(X @ w + b)

def predict_linear(X, w, b):
    """ hace la predicción """
    return X @ w + b

def compute_cost_logistic(X, y, w, b, lambda_=0, safe=False):
    """
    Calcula el costo usando la pérdida logística, versión sin matrices

    Args:
      X (ndarray): Forma (m,n)  matriz de ejemplos con n variables
      y (ndarray): Forma (m,)   valores objetivo
      w (ndarray): Forma (n,)   parámetros para la predicción
      b (scalar):               parámetro para la predicción
      lambda_ : (scalar, float) Controla la cantidad de regularización, 0 = sin regularización
      safe : (boolean)          True selecciona un algoritmo seguro contra underflow/overflow
    Returns:
      cost (scalar): costo
    """

    m,n = X.shape
    cost = 0.0
    for i in range(m):
        z_i    = np.dot(X[i],w) + b                                             #(n,)(n,) o (n,) ()
        if safe:  #evita desbordamientos
            cost += -(y[i] * z_i ) + log_1pexp(z_i)
        else:
            f_wb_i = sigmoid(z_i)                                                   #(n,)
            cost  += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)       # escalar
    cost = cost/m

    reg_cost = 0
    if lambda_ != 0:
        for j in range(n):
            reg_cost += (w[j]**2)                                               # escalar
        reg_cost = (lambda_/(2*m))*reg_cost

    return cost + reg_cost


def log_1pexp(x, maximum=20):
    ''' aproxima log(1+exp^x)
        https://stats.stackexchange.com/questions/475589/numerical-computation-of-cross-entropy-in-practice
    Args:
    x   : (ndarray de forma (n,1) o (n,))   entrada
    out : (ndarray con la misma forma que x) salida ~= np.log(1+exp(x))
    '''

    out  = np.zeros_like(x,dtype=float)
    i    = x <= maximum
    ni   = np.logical_not(i)

    out[i]  = np.log(1 + np.exp(x[i]))
    out[ni] = x[ni]
    return out


def compute_cost_matrix(X, y, w, b, logistic=False, lambda_=0, safe=True):
    """
    Calcula el costo usando matrices
    Args:
      X : (ndarray, forma (m,n))          matriz de ejemplos
      y : (ndarray, forma (m,) o (m,1))   valor objetivo de cada ejemplo
      w : (ndarray, forma (n,) o (n,1))   valores de los parámetros del modelo
      b : (scalar )                       valor del parámetro del modelo
      logistic: (boolean)                 lineal si es False, logística si es True
      lambda_:  (float)                   aplica regularización si es distinto de cero
      safe: (boolean)                     True usa el cálculo seguro contra desbordamiento (solo logística)
    Returns:
      total_cost: (scalar)                costo
    """
    m = X.shape[0]
    y = y.reshape(-1,1)             # asegura que sea 2D
    w = w.reshape(-1,1)             # asegura que sea 2D
    if logistic:
        if safe:  #seguro contra desbordamiento
            z = X @ w + b                                                           #(m,n)(n,1)=(m,1)
            cost = -(y * z) + log_1pexp(z)
            cost = np.sum(cost)/m                                                   # (escalar)
        else:
            f    = sigmoid(X @ w + b)                                               # (m,n)(n,1) = (m,1)
            cost = (1/m)*(np.dot(-y.T, np.log(f)) - np.dot((1-y).T, np.log(1-f)))   # (1,m)(m,1) = (1,1)
            cost = cost[0,0]                                                        # escalar
    else:
        f    = X @ w + b                                                        # (m,n)(n,1) = (m,1)
        cost = (1/(2*m)) * np.sum((f - y)**2)                                   # escalar

    reg_cost = (lambda_/(2*m)) * np.sum(w**2)                                   # escalar

    total_cost = cost + reg_cost                                                # escalar

    return total_cost                                                           # escalar

def compute_gradient_matrix(X, y, w, b, logistic=False, lambda_=0):
    """
    Calcula el gradiente usando matrices

    Args:
      X : (ndarray, forma (m,n))          matriz de ejemplos
      y : (ndarray, forma (m,) o (m,1))   valor objetivo de cada ejemplo
      w : (ndarray, forma (n,) o (n,1))   valores de los parámetros del modelo
      b : (scalar )                       valor del parámetro del modelo
      logistic: (boolean)                 lineal si es False, logística si es True
      lambda_:  (float)                   aplica regularización si es distinto de cero
    Returns
      dj_dw: (array_like, forma (n,1))    El gradiente del costo con respecto a los parámetros w
      dj_db: (scalar)                     El gradiente del costo con respecto al parámetro b
    """
    m = X.shape[0]
    y = y.reshape(-1,1)             # asegura que sea 2D
    w = w.reshape(-1,1)             # asegura que sea 2D

    f_wb  = sigmoid( X @ w + b ) if logistic else  X @ w + b      # (m,n)(n,1) = (m,1)
    err   = f_wb - y                                              # (m,1)
    dj_dw = (1/m) * (X.T @ err)                                   # (n,m)(m,1) = (n,1)
    dj_db = (1/m) * np.sum(err)                                   # escalar

    dj_dw += (lambda_/m) * w        # regularización              # (n,1)

    return dj_db, dj_dw                                           # escalar, (n,1)

def gradient_descent(X, y, w_in, b_in, alpha, num_iters, logistic=False, lambda_=0, verbose=True):
    """
    Ejecuta descenso de gradiente por lotes (batch) para aprender los parámetros. Los actualiza
    dando num_iters pasos de gradiente con tasa de aprendizaje alpha

    Args:
      X (ndarray):    Forma (m,n)         matriz de ejemplos
      y (ndarray):    Forma (m,) o (m,1)  valor objetivo de cada ejemplo
      w_in (ndarray): Forma (n,) o (n,1)  valores iniciales de los parámetros del modelo
      b_in (scalar):                      valor inicial del parámetro del modelo
      logistic: (boolean)                 lineal si es False, logística si es True
      lambda_:  (float)                   aplica regularización si es distinto de cero
      alpha (float):                      tasa de aprendizaje
      num_iters (int):                    número de iteraciones del descenso de gradiente
      verbose (boolean):                  si es True, imprime el costo durante el proceso

    Returns:
      w (ndarray): Forma (n,) o (n,1)     valores actualizados de los parámetros; misma forma que la entrada
      b (scalar):                         valor actualizado del parámetro
    """
    # Lista para guardar el costo J en cada iteración, principalmente para graficar después
    J_history = []
    w = copy.deepcopy(w_in)  #evita modificar el w global dentro de la función
    b = b_in
    w = w.reshape(-1,1)      #prepara para operaciones con matrices
    y = y.reshape(-1,1)

    for i in range(num_iters):

        # Calcular el gradiente
        dj_db,dj_dw = compute_gradient_matrix(X, y, w, b, logistic, lambda_)

        # Actualizar los parámetros usando w, b, alpha y el gradiente
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Guardar el costo J en cada iteración
        if i<100000:      # evita agotar los recursos
            J_history.append( compute_cost_matrix(X, y, w, b, logistic, lambda_) )

        # Imprimir el costo 10 veces a lo largo del proceso (o en cada iteración si son menos de 10)
        if i% math.ceil(num_iters / 10) == 0:
            if verbose: print(f"Iteración {i:4d}: Costo {J_history[-1]}   ")

    return w.reshape(w_in.shape), b, J_history  #devuelve w, b finales y el historial de J para graficar

def zscore_normalize_features(X):
    """
    calcula X normalizado por columna con z-score

    Args:
      X (ndarray): Forma (m,n) datos de entrada, m ejemplos, n variables

    Returns:
      X_norm (ndarray): Forma (m,n)  entrada normalizada por columna
      mu (ndarray):     Forma (n,)   media de cada variable
      sigma (ndarray):  Forma (n,)   desviación estándar de cada variable
    """
    # calcular la media de cada columna/variable
    mu     = np.mean(X, axis=0)                 # mu tendrá forma (n,)
    # calcular la desviación estándar de cada columna/variable
    sigma  = np.std(X, axis=0)                  # sigma tendrá forma (n,)
    # elemento a elemento: restar a cada ejemplo la media de su columna y dividir entre la desviación estándar de esa columna
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma

#verificar nuestro trabajo
#from sklearn.preprocessing import scale
#scale(X_orig, axis=0, with_mean=True, with_std=True, copy=True)

######################################################
# Funciones de graficación comunes
######################################################


def plot_data(X, y, ax, pos_label="y=1", neg_label="y=0", s=80, loc='best' ):
    """ grafica datos de clasificación logística con dos ejes """
    # Encontrar los índices de los ejemplos positivos y negativos
    pos = y == 1
    neg = y == 0
    pos = pos.reshape(-1,)  #funciona con vectores y de forma (m,) o (m,1)
    neg = neg.reshape(-1,)

    # Graficar los ejemplos
    ax.scatter(X[pos, 0], X[pos, 1], marker='x', s=s, c = 'red', label=pos_label)
    ax.scatter(X[neg, 0], X[neg, 1], marker='o', s=s, label=neg_label, facecolors='none', edgecolors=dlblue, lw=3)
    ax.legend(loc=loc)

    ax.figure.canvas.toolbar_visible = False
    ax.figure.canvas.header_visible = False
    ax.figure.canvas.footer_visible = False

def plt_tumor_data(x, y, ax):
    """ grafica los datos de tumores en un eje """
    pos = y == 1
    neg = y == 0

    ax.scatter(x[pos], y[pos], marker='x', s=80, c = 'red', label="maligno")
    ax.scatter(x[neg], y[neg], marker='o', s=100, label="benigno", facecolors='none', edgecolors=dlblue,lw=3)
    ax.set_ylim(-0.175,1.1)
    ax.set_ylabel('y')
    ax.set_xlabel('Tamaño del tumor')
    ax.set_title("Regresión logística con datos categóricos")

    ax.figure.canvas.toolbar_visible = False
    ax.figure.canvas.header_visible = False
    ax.figure.canvas.footer_visible = False

# Dibuja un umbral en 0.5
def draw_vthresh(ax,x):
    """ dibuja un umbral """
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ax.fill_between([xlim[0], x], [ylim[1], ylim[1]], alpha=0.2, color=dlblue)
    ax.fill_between([x, xlim[1]], [ylim[1], ylim[1]], alpha=0.2, color=dldarkred)
    ax.annotate("z >= 0", xy= [x,0.5], xycoords='data',
                xytext=[30,5],textcoords='offset points')
    d = FancyArrowPatch(
        posA=(x, 0.5), posB=(x+3, 0.5), color=dldarkred,
        arrowstyle='simple, head_width=5, head_length=10, tail_width=0.0',
    )
    ax.add_artist(d)
    ax.annotate("z < 0", xy= [x,0.5], xycoords='data',
                 xytext=[-50,5],textcoords='offset points', ha='left')
    f = FancyArrowPatch(
        posA=(x, 0.5), posB=(x-3, 0.5), color=dlblue,
        arrowstyle='simple, head_width=5, head_length=10, tail_width=0.0',
    )
    ax.add_artist(f)
