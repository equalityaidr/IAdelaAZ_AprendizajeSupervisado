import numpy as np
import copy
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.ticker import MaxNLocator
dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0'; 
plt.style.use('./deeplearning.mplstyle')

def load_data_multi():
    data = np.loadtxt("data/ex1data2.txt", delimiter=',')
    X = data[:,:2]
    y = data[:,2]
    return X, y

##########################################################
# Funciones de graficación
##########################################################

def plt_house_x(X, y,f_wb=None, ax=None):
    ''' grafica los datos de casas con ejes '''
    if not ax:
        fig, ax = plt.subplots(1,1)
    ax.scatter(X, y, marker='x', c='r', label="Valor real")

    ax.set_title("Precios de viviendas")
    ax.set_ylabel('Precio (miles de dólares)')
    ax.set_xlabel(f'Tamaño (1000 pies²)')
    if f_wb is not None:
        ax.plot(X, f_wb,  c=dlblue, label="Nuestra predicción")
    ax.legend()
    

def mk_cost_lines(x,y,w,b, ax):
    ''' dibuja las líneas verticales del costo'''
    cstr = "costo = (1/2m)*1000*("
    ctot = 0
    label = 'costo por punto'
    for p in zip(x,y):
        f_wb_p = w*p[0]+b
        c_p = ((f_wb_p - p[1])**2)/2
        c_p_txt = c_p/1000
        ax.vlines(p[0], p[1],f_wb_p, lw=3, color=dlpurple, ls='dotted', label=label)
        label='' #solo una vez
        cxy = [p[0], p[1] + (f_wb_p-p[1])/2]
        ax.annotate(f'{c_p_txt:0.0f}', xy=cxy, xycoords='data',color=dlpurple, 
            xytext=(5, 0), textcoords='offset points')
        cstr += f"{c_p_txt:0.0f} +"
        ctot += c_p
    ctot = ctot/(len(x))
    cstr = cstr[:-1] + f") = {ctot:0.0f}"
    ax.text(0.15,0.02,cstr, transform=ax.transAxes, color=dlpurple)
    
    
def inbounds(a,b,xlim,ylim):
    xlow,xhigh = xlim
    ylow,yhigh = ylim
    ax, ay = a
    bx, by = b
    if (ax > xlow and ax < xhigh) and (bx > xlow and bx < xhigh) \
        and (ay > ylow and ay < yhigh) and (by > ylow and by < yhigh):
        return(True)
    else:
        return(False)

from mpl_toolkits.mplot3d import axes3d
def plt_contour_wgrad(x, y, hist, ax, w_range=[-100, 500, 5], b_range=[-500, 500, 5], 
                contours = [0.1,50,1000,5000,10000,25000,50000], 
                      resolution=5, w_final=200, b_final=100,step=10 ):
    b0,w0 = np.meshgrid(np.arange(*b_range),np.arange(*w_range))
    z=np.zeros_like(b0)
    n,_ = w0.shape
    for i in range(w0.shape[0]):
        for j in range(w0.shape[1]):
            z[i][j] = compute_cost(x, y, w0[i][j], b0[i][j] )
   
    CS = ax.contour(w0, b0, z, contours, linewidths=2,
                   colors=[dlblue, dlorange, dldarkred, dlmagenta, dlpurple]) 
    ax.clabel(CS, inline=1, fmt='%1.0f', fontsize=10)
    ax.set_xlabel("w");  ax.set_ylabel("b")
    ax.set_title('Gráfico de contorno del costo J(w,b) vs. b,w, con la trayectoria del descenso de gradiente')
    w = w_final; b=b_final
    ax.hlines(b, ax.get_xlim()[0],w, lw=2, color=dlpurple, ls='dotted')
    ax.vlines(w, ax.get_ylim()[0],b, lw=2, color=dlpurple, ls='dotted')

    base = hist[0]
    for point in hist[0::step]:
        edist = np.sqrt((base[0] - point[0])**2 + (base[1] - point[1])**2)
        if(edist > resolution or point==hist[-1]):
            if inbounds(point,base, ax.get_xlim(),ax.get_ylim()):
                plt.annotate('', xy=point, xytext=base,xycoords='data',
                         arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 3},
                         va='center', ha='center')
            base=point
    return


# grafica p1 vs. p2. prange es un array de entradas [mín, máx, pasos]. Se usa en el laboratorio de escalado de variables.
def plt_contour_multi(x, y, w, b, ax, prange, p1, p2, title="", xlabel="", ylabel=""): 
    contours = [1e2, 2e2,3e2,4e2, 5e2, 6e2, 7e2,8e2,1e3, 1.25e3,1.5e3, 1e4, 1e5, 1e6, 1e7]
    px,py = np.meshgrid(np.linspace(*(prange[p1])),np.linspace(*(prange[p2])))
    z=np.zeros_like(px)
    n,_ = px.shape
    for i in range(px.shape[0]):
        for j in range(px.shape[1]):
            w_ij = w
            b_ij = b
            if p1 <= 3: w_ij[p1] = px[i,j]
            if p1 == 4: b_ij = px[i,j]
            if p2 <= 3: w_ij[p2] = py[i,j]
            if p2 == 4: b_ij = py[i,j]
                
            z[i][j] = compute_cost(x, y, w_ij, b_ij )
    CS = ax.contour(px, py, z, contours, linewidths=2,
                   colors=[dlblue, dlorange, dldarkred, dlmagenta, dlpurple]) 
    ax.clabel(CS, inline=1, fmt='%1.2e', fontsize=10)
    ax.set_xlabel(xlabel);  ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14)


def plt_equal_scale(X_train, X_norm, y_train):
    fig,ax = plt.subplots(1,2,figsize=(12,5))
    prange = [
              [ 0.238-0.045, 0.238+0.045,  50],
              [-25.77326319-0.045, -25.77326319+0.045, 50],
              [-50000, 0,      50],
              [-1500,  0,      50],
              [0, 200000, 50]]
    w_best = np.array([0.23844318, -25.77326319, -58.11084634,  -1.57727192])
    b_best = 235
    plt_contour_multi(X_train, y_train, w_best, b_best, ax[0], prange, 0, 1, 
                      title='Sin normalizar, J(w,b) vs. w[0],w[1]',
                      xlabel= "w[0] (tamaño en pies²)", ylabel="w[1] (n.º de habitaciones)")
    #
    w_best = np.array([111.1972, -16.75480051, -28.51530411, -37.17305735])
    b_best = 376.949151515151
    prange = [[ 111-50, 111+50,   75],
              [-16.75-50,-16.75+50, 75],
              [-28.5-8, -28.5+8,  50],
              [-37.1-16,-37.1+16, 50],
              [376-150, 376+150, 50]]
    plt_contour_multi(X_norm, y_train, w_best, b_best, ax[1], prange, 0, 1, 
                      title='Normalizado, J(w,b) vs. w[0],w[1]',
                      xlabel= "w[0] (tamaño normalizado)", ylabel="w[1] (n.º de habitaciones normalizado)")
    fig.suptitle("Contorno del costo con la misma escala", fontsize=18)
    #plt.tight_layout(rect=(0,0,1.05,1.05))
    fig.tight_layout(rect=(0,0,1,0.95))
    plt.show()
    
def plt_divergence(p_hist, J_hist, x_train,y_train):

    x=np.zeros(len(p_hist))
    y=np.zeros(len(p_hist))
    v=np.zeros(len(p_hist))
    for i in range(len(p_hist)):
        x[i] = p_hist[i][0]
        y[i] = p_hist[i][1]
        v[i] = J_hist[i]

    fig = plt.figure(figsize=(12,5))
    plt.subplots_adjust( wspace=0 )
    gs = fig.add_gridspec(1, 5)
    fig.suptitle(f"El costo se dispara cuando la tasa de aprendizaje es demasiado grande")
    #===============
    #  Primera subgráfica
    #===============
    ax = fig.add_subplot(gs[:2], )

    # Graficar w vs. costo para ver el mínimo
    fix_b = 100
    w_array = np.arange(-70000, 70000, 1000)
    cost = np.zeros_like(w_array)

    for i in range(len(w_array)):
        tmp_w = w_array[i]
        cost[i] = compute_cost(x_train, y_train, tmp_w, fix_b)

    ax.plot(w_array, cost)
    ax.plot(x,v, c=dlmagenta)
    ax.set_title("Costo vs. w, con b fijo en 100")
    ax.set_ylabel('Costo')
    ax.set_xlabel('w')
    ax.xaxis.set_major_locator(MaxNLocator(2)) 

    #===============
    # Segunda subgráfica
    #===============

    tmp_b,tmp_w = np.meshgrid(np.arange(-35000, 35000, 500),np.arange(-70000, 70000, 500))
    z=np.zeros_like(tmp_b)
    for i in range(tmp_w.shape[0]):
        for j in range(tmp_w.shape[1]):
            z[i][j] = compute_cost(x_train, y_train, tmp_w[i][j], tmp_b[i][j] )

    ax = fig.add_subplot(gs[2:], projection='3d')
    ax.plot_surface(tmp_w, tmp_b, z,  alpha=0.3, color=dlblue)
    ax.xaxis.set_major_locator(MaxNLocator(2)) 
    ax.yaxis.set_major_locator(MaxNLocator(2)) 

    ax.set_xlabel('w', fontsize=16)
    ax.set_ylabel('b', fontsize=16)
    ax.set_zlabel('\ncosto', fontsize=16)
    plt.title('Costo vs. (b, w)')
    # Ajustar el ángulo de visión 
    ax.view_init(elev=20., azim=-65)
    ax.plot(x, y, v,c=dlmagenta)
    
    return

# dibujar la recta de la derivada
# y = m*(x - x1) + y1
def add_line(dj_dx, x1, y1, d, ax):
    x = np.linspace(x1-d, x1+d,50)
    y = dj_dx*(x - x1) + y1
    ax.scatter(x1, y1, color=dlblue, s=50)
    ax.plot(x, y, '--', c=dldarkred,zorder=10, linewidth = 1)
    xoff = 30 if x1 == 200 else 10
    ax.annotate(r"$\frac{\partial J}{\partial w}$ =%d" % dj_dx, fontsize=14,
                xy=(x1, y1), xycoords='data',
            xytext=(xoff, 10), textcoords='offset points',
            arrowprops=dict(arrowstyle="->"),
            horizontalalignment='left', verticalalignment='top')

def plt_gradients(x_train,y_train, f_compute_cost, f_compute_gradient):
    #===============
    #  Primera subgráfica
    #===============
    fig,ax = plt.subplots(1,2,figsize=(12,4))

    # Graficar w vs. costo para ver el mínimo
    fix_b = 100
    w_array = np.linspace(-100, 500, 50)
    w_array = np.linspace(0, 400, 50)
    cost = np.zeros_like(w_array)

    for i in range(len(w_array)):
        tmp_w = w_array[i]
        cost[i] = f_compute_cost(x_train, y_train, tmp_w, fix_b)
    ax[0].plot(w_array, cost,linewidth=1)
    ax[0].set_title("Costo vs. w, con el gradiente; b fijo en 100")
    ax[0].set_ylabel('Costo')
    ax[0].set_xlabel('w')

    # dibujar las rectas con b fijo en 100
    for tmp_w in [100,200,300]:
        fix_b = 100
        dj_dw,dj_db = f_compute_gradient(x_train, y_train, tmp_w, fix_b )
        j = f_compute_cost(x_train, y_train, tmp_w, fix_b)
        add_line(dj_dw, tmp_w, j, 30, ax[0])

    #===============
    # Segunda subgráfica
    #===============

    tmp_b,tmp_w = np.meshgrid(np.linspace(-200, 200, 10), np.linspace(-100, 600, 10))
    U = np.zeros_like(tmp_w)
    V = np.zeros_like(tmp_b)
    for i in range(tmp_w.shape[0]):
        for j in range(tmp_w.shape[1]):
            U[i][j], V[i][j] = f_compute_gradient(x_train, y_train, tmp_w[i][j], tmp_b[i][j] )
    X = tmp_w
    Y = tmp_b
    n=-2
    color_array = np.sqrt(((V-n)/2)**2 + ((U-n)/2)**2)

    ax[1].set_title('Gradiente en un gráfico de flechas (quiver plot)')
    Q = ax[1].quiver(X, Y, U, V, color_array, units='width', )
    qk = ax[1].quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',coordinates='figure')
    ax[1].set_xlabel("w"); ax[1].set_ylabel("b")

def norm_plot(ax, data):
    scale = (np.max(data) - np.min(data))*0.2
    x = np.linspace(np.min(data)-scale,np.max(data)+scale,50)
    _,bins, _ = ax.hist(data, x, color="xkcd:azure")
    #ax.set_ylabel("Frecuencia")
    
    mu = np.mean(data); 
    std = np.std(data); 
    dist = norm.pdf(bins, loc=mu, scale = std)
    
    axr = ax.twinx()
    axr.plot(bins,dist, color = "orangered", lw=2)
    axr.set_ylim(bottom=0)
    axr.axis('off')
    
def plot_cost_i_w(X,y,hist):
    ws = np.array([ p[0] for p in hist["params"]])
    rng = max(abs(ws[:,0].min()),abs(ws[:,0].max()))
    wr = np.linspace(-rng+0.27,rng+0.27,20)
    cst = [compute_cost(X,y,np.array([wr[i],-32, -67, -1.46]), 221) for i in range(len(wr))]

    fig,ax = plt.subplots(1,2,figsize=(12,3))
    ax[0].plot(hist["iter"], (hist["cost"]));  ax[0].set_title("Costo vs. iteración")
    ax[0].set_xlabel("Iteración"); ax[0].set_ylabel("Costo")
    ax[1].plot(wr, cst); ax[1].set_title("Costo vs. w[0]")
    ax[1].set_xlabel("w[0]"); ax[1].set_ylabel("Costo")
    ax[1].plot(ws[:,0],hist["cost"])
    plt.show()

 
##########################################################
# Funciones de regresión
##########################################################

def compute_gradient_matrix(X, y, w, b): 
    """
    Calcula el gradiente para regresión lineal 
 
    Args:
      X : (array_like, forma (m,n)) variable, por ejemplo, el tamaño de la casa 
      y : (array_like, forma (m,1)) valor real 
      w : (array_like, forma (n,1)) valores de los parámetros del modelo      
      b : (scalar )                 valor del parámetro del modelo      
    Returns
      dj_dw: (array_like, forma (n,1)) El gradiente del costo con respecto a los parámetros w. 
      dj_db: (scalar)                  El gradiente del costo con respecto al parámetro b. 
                                  
    """
    m,n = X.shape
    f_wb = X @ w + b              
    e   = f_wb - y                
    dj_dw  = (1/m) * (X.T @ e)    
    dj_db  = (1/m) * np.sum(e)    
        
    return dj_db,dj_dw

#Función para calcular el costo
def compute_cost_matrix(X, y, w, b, verbose=False):
    """
    Calcula el costo para regresión lineal 
     Args:
      X : (array_like, forma (m,n)) variable, por ejemplo, el tamaño de la casa 
      y : (array_like, forma (m,))  valor real 
      w : (array_like, forma (n,))  parámetros del modelo 
      b : (scalar               )   parámetro del modelo 
      verbose : (Boolean) Si es True, imprime el valor intermedio f_wb
    Returns
      cost: (scalar)                      
    """ 
    m,n = X.shape

    # calcular f_wb para todos los ejemplos
    f_wb = X @ w + b  
    # calcular el costo
    total_cost = (1/(2*m)) * np.sum((f_wb-y)**2)

    if verbose: print("f_wb:")
    if verbose: print(f_wb)
        
    return total_cost

# Versión con ciclo de compute_cost para múltiples variables
def compute_cost(X, y, w, b): 
    """
    calcula el costo
    Args:
      X : (ndarray): Forma (m,n) matriz de ejemplos con múltiples variables
      y : (ndarray): Forma (m,)  valores objetivo
      w : (ndarray): Forma (n)   parámetros para la predicción   
      b : (scalar):              parámetro para la predicción   
    Returns
      cost: (scalar)             costo
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i],w) + b       
        cost = cost + (f_wb_i - y[i])**2              
    cost = cost/(2*m)                                 
    return(np.squeeze(cost)) 

def compute_gradient(X, y, w, b): 
    """
    Calcula el gradiente para regresión lineal 
    Args:
      X : (ndarray, forma (m,n)) matriz de ejemplos 
      y : (ndarray, forma (m,))  valor objetivo de cada ejemplo
      w : (ndarray, forma (n,))  parámetros del modelo      
      b : (scalar)               parámetro del modelo      
    Returns
      dj_dw : (ndarray, forma (n,)) El gradiente del costo con respecto a los parámetros w. 
      dj_db : (scalar)              El gradiente del costo con respecto al parámetro b. 
    """
    m,n = X.shape           #(número de ejemplos, número de variables)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i,j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw/m                                
    dj_db = dj_db/m                                
        
    return dj_db,dj_dw

#Esta versión guarda más valores e imprime más información que las versiones de los ejercicios
def gradient_descent_houses(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Ejecuta descenso de gradiente por lotes (batch) para aprender los parámetros. Los actualiza
    dando num_iters pasos de gradiente con tasa de aprendizaje alpha
    
    Args:
      X : (array_like, forma (m,n))    matriz de ejemplos 
      y : (array_like, forma (m,))     valor objetivo de cada ejemplo
      w_in : (array_like, forma (n,))  valores iniciales de los parámetros del modelo
      b_in : (scalar)                  valor inicial del parámetro del modelo
      cost_function: función para calcular el costo
      gradient_function: función para calcular el gradiente
      alpha : (float) tasa de aprendizaje
      num_iters : (int) número de iteraciones del descenso de gradiente
    Returns
      w : (array_like, forma (n,)) valores actualizados de los parámetros del modelo
          después del descenso de gradiente
      b : (scalar)                 valor actualizado del parámetro del modelo
          después del descenso de gradiente
      hist : (dict)                historial de costo, parámetros, gradientes e iteraciones
    """
    
    # número de ejemplos de entrenamiento
    m = len(X)
    
    # Diccionario para guardar valores en cada iteración, principalmente para graficar después
    hist={}
    hist["cost"] = []; hist["params"] = []; hist["grads"]=[]; hist["iter"]=[];
    
    w = copy.deepcopy(w_in)  #evita modificar el w global dentro de la función
    b = b_in
    save_interval = np.ceil(num_iters/10000) # evita agotar los recursos en ejecuciones largas

    print(f"Iteración Costo         w0       w1       w2       w3       b       djdw0    djdw1    djdw2    djdw3    djdb  ")
    print(f"---------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|")

    for i in range(num_iters):

        # Calcular el gradiente
        dj_db,dj_dw = gradient_function(X, y, w, b)   

        # Actualizar los parámetros usando w, b, alpha y el gradiente
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
        # Guardar el costo J, w y b en cada intervalo de guardado, para graficar
        if i == 0 or i % save_interval == 0:     
            hist["cost"].append(cost_function(X, y, w, b))
            hist["params"].append([w,b])
            hist["grads"].append([dj_dw,dj_db])
            hist["iter"].append(i)

        # Imprimir el costo 10 veces a lo largo del proceso (o en cada iteración si son menos de 10)
        if i% math.ceil(num_iters/10) == 0:
            #print(f"Iteración {i:4d}: Costo {cost_function(X, y, w, b):8.2f}   ")
            cst = cost_function(X, y, w, b)
            print(f"{i:9d} {cst:0.5e} {w[0]: 0.1e} {w[1]: 0.1e} {w[2]: 0.1e} {w[3]: 0.1e} {b: 0.1e} {dj_dw[0]: 0.1e} {dj_dw[1]: 0.1e} {dj_dw[2]: 0.1e} {dj_dw[3]: 0.1e} {dj_db: 0.1e}")
       
    return w, b, hist #devuelve w, b y el historial para graficar

def run_gradient_descent(X,y,iterations=1000, alpha = 1e-6):

    m,n = X.shape
    # inicializar los parámetros
    initial_w = np.zeros(n)
    initial_b = 0
    # ejecutar el descenso de gradiente
    w_out, b_out, hist_out = gradient_descent_houses(X ,y, initial_w, initial_b,
                                               compute_cost, compute_gradient_matrix, alpha, iterations)
    print(f"w,b encontrados por descenso de gradiente: w: {w_out}, b: {b_out:0.2f}")
    
    return(w_out, b_out, hist_out)

# extracción compacta de los datos de hist
#x = hist["iter"]
#J  = np.array([ p    for p in hist["cost"]])
#ws = np.array([ p[0] for p in hist["params"]])
#dj_ws = np.array([ p[0] for p in hist["grads"]])

#bs = np.array([ p[1] for p in hist["params"]]) 

def run_gradient_descent_feng(X,y,iterations=1000, alpha = 1e-6):
    m,n = X.shape
    # inicializar los parámetros
    initial_w = np.zeros(n)
    initial_b = 0
    # ejecutar el descenso de gradiente
    w_out, b_out, hist_out = gradient_descent(X ,y, initial_w, initial_b,
                                               compute_cost, compute_gradient_matrix, alpha, iterations)
    print(f"w,b encontrados por descenso de gradiente: w: {w_out}, b: {b_out:0.4f}")
    
    return(w_out, b_out)

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Ejecuta descenso de gradiente por lotes (batch) para aprender los parámetros. Los actualiza
    dando num_iters pasos de gradiente con tasa de aprendizaje alpha
    
    Args:
      X : (array_like, forma (m,n))    matriz de ejemplos 
      y : (array_like, forma (m,))     valor objetivo de cada ejemplo
      w_in : (array_like, forma (n,))  valores iniciales de los parámetros del modelo
      b_in : (scalar)                  valor inicial del parámetro del modelo
      cost_function: función para calcular el costo
      gradient_function: función para calcular el gradiente
      alpha : (float) tasa de aprendizaje
      num_iters : (int) número de iteraciones del descenso de gradiente
    Returns
      w : (array_like, forma (n,)) valores actualizados de los parámetros del modelo
          después del descenso de gradiente
      b : (scalar)                 valor actualizado del parámetro del modelo
          después del descenso de gradiente
      hist : (dict)                historial de costo, parámetros, gradientes e iteraciones
    """
    
    # número de ejemplos de entrenamiento
    m = len(X)
    
    # Diccionario para guardar valores en cada iteración, principalmente para graficar después
    hist={}
    hist["cost"] = []; hist["params"] = []; hist["grads"]=[]; hist["iter"]=[];
    
    w = copy.deepcopy(w_in)  #evita modificar el w global dentro de la función
    b = b_in
    save_interval = np.ceil(num_iters/10000) # evita agotar los recursos en ejecuciones largas

    for i in range(num_iters):

        # Calcular el gradiente
        dj_db,dj_dw = gradient_function(X, y, w, b)   

        # Actualizar los parámetros usando w, b, alpha y el gradiente
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
        # Guardar el costo J, w y b en cada intervalo de guardado, para graficar
        if i == 0 or i % save_interval == 0:     
            hist["cost"].append(cost_function(X, y, w, b))
            hist["params"].append([w,b])
            hist["grads"].append([dj_dw,dj_db])
            hist["iter"].append(i)

        # Imprimir el costo 10 veces a lo largo del proceso (o en cada iteración si son menos de 10)
        if i% math.ceil(num_iters/10) == 0:
            #print(f"Iteración {i:4d}: Costo {cost_function(X, y, w, b):8.2f}   ")
            cst = cost_function(X, y, w, b)
            print(f"Iteración {i:9d}, Costo: {cst:0.5e}")
    return w, b, hist #devuelve w, b y el historial para graficar

def load_house_data():
    data = np.loadtxt("./data/houses.txt", delimiter=',', skiprows=1)
    X = data[:,:4]
    y = data[:,4]
    return X, y

def zscore_normalize_features(X,rtn_ms=False):
    """
    devuelve X normalizado por columna con z-score
    Args:
      X : (numpy array (m,n)) 
      rtn_ms : (boolean) si es True, también devuelve la media (mu) y la desviación estándar (sigma)
    Returns
      X_norm: (numpy array (m,n)) entrada normalizada por columna
      mu, sigma: (numpy array (n,)) solo si rtn_ms es True
    """
    mu     = np.mean(X,axis=0)  
    sigma  = np.std(X,axis=0)
    X_norm = (X - mu)/sigma      

    if rtn_ms:
        return(X_norm, mu, sigma)
    else:
        return(X_norm)
