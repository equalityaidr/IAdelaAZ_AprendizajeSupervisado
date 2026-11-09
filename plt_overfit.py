"""
plot_overfit
    clase y funciones asociadas que grafican un ejemplo interactivo de sobreajuste (overfitting) y sus soluciones
"""
import math
from ipywidgets import Output
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, CheckButtons
from sklearn.linear_model import LogisticRegression, Ridge
from lab_utils_common import np, plt, dlc, predict_logistic, plot_data, zscore_normalize_features

def map_one_feature(X1, degree):
    """
    Transforma una variable en variables polinomiales
    """
    X1 = np.atleast_1d(X1)
    out = []
    string = ""
    k = 0
    for i in range(1, degree+1):
        out.append((X1**i))
        string = string + f"w_{{{k}}}{munge('x_0',i)} + "
        k += 1
    string = string + ' b' #se agrega b al texto de la ecuación, no a los datos
    return np.stack(out, axis=1), string


def map_feature(X1, X2, degree):
    """
    Transforma dos variables en variables polinomiales
    """
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)

    out = []
    string = ""
    k = 0
    for i in range(1, degree+1):
        for j in range(i + 1):
            out.append((X1**(i-j) * (X2**j)))
            string = string + f"w_{{{k}}}{munge('x_0',i-j)}{munge('x_1',j)} + "
            k += 1
    #print(string + 'b')
    return np.stack(out, axis=1), string + ' b'

def munge(base, exp):
    if exp == 0:
        return ''
    if exp == 1:
        return base
    return base + f'^{{{exp}}}'

def plot_decision_boundary(ax, x0r,x1r, predict,  w, b, scaler = False, mu=None, sigma=None, degree=None):
    """
    Grafica una frontera de decisión
     Args:
      ax : eje de matplotlib donde se dibuja
      x0r : (array_like, forma (1,1)) rango (mín, máx) de x0
      x1r : (array_like, forma (1,1)) rango (mín, máx) de x1
      predict : función para predecir los valores de z
      w, b : parámetros del modelo
      scaler : (boolean) si se escalan o no los datos
      mu, sigma : media y desviación estándar para escalar
      degree : (int) grado del polinomio
    """

    h = .01  # tamaño del paso de la malla
    # crear una malla para graficar
    xx, yy = np.meshgrid(np.arange(x0r[0], x0r[1], h),
                         np.arange(x1r[0], x1r[1], h))

    # Graficar la frontera de decisión. Para eso, asignamos un color a cada
    # punto de la malla [x_min, x_max]x[y_min, y_max].
    points = np.c_[xx.ravel(), yy.ravel()]
    Xm,_ = map_feature(points[:, 0], points[:, 1],degree)
    if scaler:
        Xm = (Xm - mu)/sigma
    Z = predict(Xm, w, b)

    # Mostrar el resultado como contorno
    Z = Z.reshape(xx.shape)
    contour = ax.contour(xx, yy, Z, levels = [0.5], colors='g')
    return contour

# usar esta función para probar la anterior
def plot_decision_boundary_sklearn(x0r, x1r, predict, degree,  scaler = False):
    """
    Grafica una frontera de decisión
     Args:
      x0r : (array_like, forma (1,1)) rango (mín, máx) de x0
      x1r : (array_like, forma (1,1)) rango (mín, máx) de x1
      degree: (int)                   grado del polinomio
      predict : función para predecir los valores de z
      scaler  : escalador de scikit-learn ya ajustado, o False para no escalar
    """

    h = .01  # tamaño del paso de la malla
    # crear una malla para graficar
    xx, yy = np.meshgrid(np.arange(x0r[0], x0r[1], h),
                         np.arange(x1r[0], x1r[1], h))

    # Graficar la frontera de decisión. Para eso, asignamos un color a cada
    # punto de la malla [x_min, x_max]x[y_min, y_max].
    points = np.c_[xx.ravel(), yy.ravel()]
    Xm, _ = map_feature(points[:, 0], points[:, 1],degree)
    if scaler:
        Xm = scaler.transform(Xm)
    Z = predict(Xm)

    # Mostrar el resultado como contorno
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, colors='g')
    #plot_data(X_train,y_train)

#para depurar, descomenta las líneas #@output de las funciones de las que quieras ver los errores
# En el notebook que llama a estas funciones, importa `output`:
# from plt_overfit import overfit_example, output
# luego, en la celda donde quieras ver los mensajes de error:
#display(output)

output = Output() # muestra los mensajes de error ocultos cuando se usan widgets

class button_manager:
    ''' Suple algunas funciones que les faltan a las casillas de verificación de matplotlib
    al iniciar:
        crea las casillas, las conecta con button_click
        y llama a call_on_click con el índice activo y firsttime=True
    al hacer clic:
        mantiene una sola casilla activa y llama a call_on_click
    '''

    @output.capture()  # depuración
    def __init__(self,fig, dim, labels, init, call_on_click):
        '''
        dim: (list)     [x_izquierda, y_abajo, ancho, alto]
        labels: (list)  por ejemplo ['1','2','3','4','5','6']
        init: (list)    por ejemplo [True, False, False, False, False, False]
        '''
        self.fig = fig
        self.ax = plt.axes(dim)  #izq., abajo, ancho, alto
        self.init_state = init
        self.call_on_click = call_on_click
        self.button  = CheckButtons(self.ax,labels,init)
        self.button.on_clicked(self.button_click)
        self.status = self.button.get_status()
        self.call_on_click(self.status.index(True),firsttime=True)

    @output.capture()  # depuración
    def reinit(self):
        self.status = self.init_state
        self.button.set_active(self.status.index(True))      #desactiva la anterior; esto dispara la actualización

    @output.capture()  # depuración
    def button_click(self, event):
        ''' mantiene una sola casilla activa. Si se hace clic en la casilla activa, también funciona bien '''
        #new_status = self.button.get_status()
        #new = [self.status[i] ^ new_status[i] for i in range(len(self.status))]
        #newidx = new.index(True)
        self.button.eventson = False
        self.button.set_active(self.status.index(True))  #desactiva la anterior o la reactiva si es la misma
        self.button.eventson = True
        self.status = self.button.get_status()
        self.call_on_click(self.status.index(True))

class overfit_example():
    """ grafica el ejemplo de sobreajuste """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-locals
    # pylint: disable=missing-function-docstring
    # pylint: disable=attribute-defined-outside-init
    def __init__(self, regularize=False):
        self.regularize=regularize
        self.lambda_=0
        fig = plt.figure( figsize=(8,6))
        fig.canvas.toolbar_visible = False
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.set_facecolor('#ffffff') #blanco
        gs  = GridSpec(5, 3, figure=fig)
        ax0 = fig.add_subplot(gs[0:3, :])
        ax1 = fig.add_subplot(gs[-2, :])
        ax2 = fig.add_subplot(gs[-1, :])
        ax1.set_axis_off()
        ax2.set_axis_off()
        self.ax = [ax0,ax1,ax2]
        self.fig = fig

        self.axfitdata = plt.axes([0.26,0.124,0.12,0.1 ])  #izq., abajo, ancho, alto
        self.bfitdata  = Button(self.axfitdata , 'ajustar\ndatos', color=dlc['dlblue'])
        self.bfitdata.label.set_fontsize(12)
        self.bfitdata.on_clicked(self.fitdata_clicked)

        #borrar datos es una mejora a futuro
        #self.axclrdata = plt.axes([0.26,0.06,0.12,0.05 ])  #izq., abajo, ancho, alto
        #self.bclrdata  = Button(self.axclrdata , 'borrar datos', color='white')
        #self.bclrdata.label.set_fontsize(12)
        #self.bclrdata.on_clicked(self.clrdata_clicked)

        self.cid = fig.canvas.mpl_connect('button_press_event', self.add_data)

        self.typebut = button_manager(fig, [0.4, 0.07,0.15,0.15], ["Regresión", "Categórica"],
                                       [False,True], self.toggle_type)

        self.fig.text(0.1, 0.02+0.21, "Grado", fontsize=12)
        self.degrbut = button_manager(fig,[0.1,0.02,0.15,0.2 ], ['1','2','3','4','5','6'],
                                        [True, False, False, False, False, False], self.update_equation)
        if self.regularize:
            self.fig.text(0.6, 0.02+0.21, r"lambda($\lambda$)", fontsize=12)
            self.lambut = button_manager(fig,[0.6,0.02,0.15,0.2 ], ['0.0','0.2','0.4','0.6','0.8','1'],
                                        [True, False, False, False, False, False], self.updt_lambda)

        #self.regbut =  button_manager(fig, [0.8, 0.08,0.24,0.15], ["Regularizar"],
        #                               [False], self.toggle_reg)
        #self.logistic_data()

    def updt_lambda(self, idx, firsttime=False):
      # pylint: disable=unused-argument
        self.lambda_ = idx * 0.2

    def toggle_type(self, idx, firsttime=False):
        self.logistic = idx==1
        self.ax[0].clear()
        if self.logistic:
            self.logistic_data()
        else:
            self.linear_data()
        if not firsttime:
            self.degrbut.reinit()

    @output.capture()  # depuración
    def logistic_data(self,redraw=False):
        if not redraw:
            m = 50
            n = 2
            np.random.seed(2)
            X_train = 2*(np.random.rand(m,n)-[0.5,0.5])
            y_train = X_train[:,1]+0.5  > X_train[:,0]**2 + 0.5*np.random.rand(m) #cuadrática + ruido
            y_train = y_train + 0  #convierte de booleano a entero
            self.X = X_train
            self.y = y_train
            self.x_ideal = np.sort(X_train[:,0])
            self.y_ideal =  self.x_ideal**2


        self.ax[0].plot(self.x_ideal, self.y_ideal, "--", color = "orangered", label="ideal", lw=1)
        plot_data(self.X, self.y, self.ax[0], s=10, loc='lower right')
        self.ax[0].set_title("Ejemplo de sobreajuste: datos categóricos con ruido")
        self.ax[0].text(0.5,0.93, "Haz clic para agregar datos. Con [Shift] agregas datos azules (y=0).",
                        fontsize=12, ha='center',transform=self.ax[0].transAxes, color=dlc["dlblue"])
        self.ax[0].set_xlabel(r"$x_0$")
        self.ax[0].set_ylabel(r"$x_1$")

    def linear_data(self,redraw=False):
        if not redraw:
            m = 30
            c = 0
            x_train = np.arange(0,m,1)
            np.random.seed(1)
            y_ideal = x_train**2 + c
            y_train = y_ideal + 0.7 * y_ideal*(np.random.sample((m,))-0.5)
            self.x_ideal = x_train #para volver a dibujar cuando se agregan datos nuevos a X
            self.X = x_train
            self.y = y_train
            self.y_ideal = y_ideal
        else:
            self.ax[0].set_xlim(self.xlim)
            self.ax[0].set_ylim(self.ylim)

        self.ax[0].scatter(self.X,self.y, label="y")
        self.ax[0].plot(self.x_ideal, self.y_ideal, "--", color = "orangered", label="y_ideal", lw=1)
        self.ax[0].set_title("Ejemplo de sobreajuste: datos de regresión (cuadrática con ruido)",fontsize = 14)
        self.ax[0].set_xlabel("x")
        self.ax[0].set_ylabel("y")
        self.ax0ledgend = self.ax[0].legend(loc='lower right')
        self.ax[0].text(0.5,0.93, "Haz clic en la gráfica para agregar datos",
                        fontsize=12, ha='center',transform=self.ax[0].transAxes, color=dlc["dlblue"])
        if not redraw:
            self.xlim = self.ax[0].get_xlim()
            self.ylim = self.ax[0].get_ylim()


    @output.capture()  # depuración
    def add_data(self, event):
        if self.logistic:
            self.add_data_logistic(event)
        else:
            self.add_data_linear(event)

    @output.capture()  # depuración
    def add_data_logistic(self, event):
        if event.inaxes == self.ax[0]:
            x0_coord = event.xdata
            x1_coord = event.ydata

            if event.key is None:  #no se presionó shift
                self.ax[0].scatter(x0_coord, x1_coord, marker='x', s=10, c = 'red', label="y=1")
                self.y = np.append(self.y,1)
            else:
                self.ax[0].scatter(x0_coord, x1_coord, marker='o', s=10, label="y=0", facecolors='none',
                                   edgecolors=dlc['dlblue'],lw=3)
                self.y = np.append(self.y,0)
            self.X = np.append(self.X,np.array([[x0_coord, x1_coord]]),axis=0)
        self.fig.canvas.draw()

    def add_data_linear(self, event):
        if event.inaxes == self.ax[0]:
            x_coord = event.xdata
            y_coord = event.ydata

            self.ax[0].scatter(x_coord, y_coord, marker='o', s=10, facecolors='none',
                                   edgecolors=dlc['dlblue'],lw=3)
            self.y = np.append(self.y,y_coord)
            self.X = np.append(self.X,x_coord)
            self.fig.canvas.draw()

    #@output.capture()  # depuración
    #def clrdata_clicked(self,event):
    #    if self.logistic == True:
    #        self.X = np.
    #    else:
    #        self.linear_regression()


    @output.capture()  # depuración
    def fitdata_clicked(self,event):
        if self.logistic:
            self.logistic_regression()
        else:
            self.linear_regression()

    def linear_regression(self):
        self.ax[0].clear()
        self.fig.canvas.draw()

        # crear y ajustar el modelo con el conjunto de variables transformadas (X_mapped)
        self.X_mapped, _ =  map_one_feature(self.X, self.degree)
        self.X_mapped_scaled, self.X_mu, self.X_sigma  = zscore_normalize_features(self.X_mapped)

        #linear_model = LinearRegression()
        # El original usaba Ridge(alpha=lambda_, normalize=True), pero scikit-learn eliminó normalize
        # en la versión 1.2. Como X ya está normalizado con z-score, multiplicar alpha por m
        # da el mismo nivel de regularización que tenía el original.
        linear_model = Ridge(alpha=self.lambda_ * len(self.y), max_iter=10000)
        linear_model.fit(self.X_mapped_scaled, self.y )
        self.w = linear_model.coef_.reshape(-1,)
        self.b = linear_model.intercept_
        x = np.linspace(*self.xlim,30)  #la línea se grafica independientemente de los datos, que pueden quedar desordenados
        xm, _ =  map_one_feature(x, self.degree)
        xms = (xm - self.X_mu)/ self.X_sigma
        y_pred = linear_model.predict(xms)

        #self.fig.canvas.draw()
        self.linear_data(redraw=True)
        self.ax0yfit = self.ax[0].plot(x, y_pred, color = "blue", label="y_ajustada")
        self.ax0ledgend = self.ax[0].legend(loc='lower right')
        self.fig.canvas.draw()

    def logistic_regression(self):
        self.ax[0].clear()
        self.fig.canvas.draw()

        # crear y ajustar el modelo con el conjunto de variables transformadas (X_mapped)
        self.X_mapped, _ =  map_feature(self.X[:, 0], self.X[:, 1], self.degree)
        self.X_mapped_scaled, self.X_mu, self.X_sigma  = zscore_normalize_features(self.X_mapped)
        if not self.regularize or self.lambda_ == 0:
            # Sin regularización. El original usaba penalty='none', que scikit-learn ya no acepta;
            # C=np.inf produce el mismo resultado en versiones nuevas y antiguas.
            lr = LogisticRegression(C=np.inf, max_iter=10000)
        else:
            C = 1/self.lambda_
            lr = LogisticRegression(C=C, max_iter=10000)

        lr.fit(self.X_mapped_scaled,self.y)
        #print(lr.score(self.X_mapped_scaled, self.y))
        self.w = lr.coef_.reshape(-1,)
        self.b = lr.intercept_
        #print(self.w, self.b)
        self.logistic_data(redraw=True)
        self.contour = plot_decision_boundary(self.ax[0],[-1,1],[-1,1], predict_logistic, self.w, self.b,
                       scaler=True, mu=self.X_mu, sigma=self.X_sigma, degree=self.degree )
        self.fig.canvas.draw()

    @output.capture()  # depuración
    def update_equation(self, idx, firsttime=False):
        #print(f"Actualizar ecuación, índice = {idx}, firsttime={firsttime}")
        self.degree = idx+1
        if firsttime:
            self.eqtext = []
        else:
            for artist in self.eqtext:
                #print(artist)
                artist.remove()
            self.eqtext = []
        if self.logistic:
            _, equation =  map_feature(self.X[:, 0], self.X[:, 1], self.degree)
            string = 'f_{wb} = sigmoid('
        else:
            _, equation =  map_one_feature(self.X, self.degree)
            string = 'f_{wb} = ('
        bz = 10
        seq = equation.split('+')
        blks = math.ceil(len(seq)/bz)
        for i in range(blks):
            if i == 0:
                string = string +  '+'.join(seq[bz*i:bz*i+bz])
            else:
                string = '+'.join(seq[bz*i:bz*i+bz])
            string = string + ')' if i == blks-1 else string + '+'
            ei = self.ax[1].text(0.01,(0.75-i*0.25), f"${string}$",fontsize=9,
                                 transform = self.ax[1].transAxes, ma='left', va='top' )
            self.eqtext.append(ei)
        self.fig.canvas.draw()
