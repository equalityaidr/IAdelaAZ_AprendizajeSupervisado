

#Ejercicio de regresión logística
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data[:,:2] #Tomamos solo dos variables
y = iris.target

df= pd.DataFrame(iris.data, columns=iris.feature_names)

#Exploración de datos
print("Primeros 10 registros de variables")
print(X[:10,:])
print("--------------------------")
print("Primeros 10 valores objetivo")
print(y[:10])
print(iris.DESCR)

from pandas_profiling import ProfileReport
prof = ProfileReport(df)
prof.to_file(output_file='output.html')

#Cantidad por clase
unique, counts = np.unique(y, return_counts=True)
dict(zip(unique, counts))

#Gráfico de dispersión
plt.figure(2, figsize=(8, 6))
plt.clf()
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
legend1 = plt.legend(*scatter.legend_elements(),loc = "upper right", title="Clases")
plt.xlabel('Largo del sépalo')
plt.ylabel('Ancho del sépalo')
plt.show()

#Eliminando la clase '2'
new_y = y[:100]
new_X = X[:100]

#Gráfico de dispersión 2
plt.figure(2, figsize=(8, 6))
plt.clf()
scatter = plt.scatter(new_X[:, 0], new_X[:, 1], c=new_y,cmap=plt.cm.Set1, edgecolor='k')
legend1 = plt.legend(*scatter.legend_elements(), loc="upper right", title="Clases")
plt.xlabel('Largo del sépalo')
plt.ylabel('Ancho del sépalo')
plt.show()

#División en entrenamiento y prueba
X_train,X_test,Y_train,Y_test=train_test_split(new_X,new_y,test_size=0.2,random_state=0)
print(X_train.shape,X_test.shape)
print(Y_train.shape,Y_test.shape)

#Regresión logística
logreg = LogisticRegression(solver='lbfgs', C=0.12)
logreg.fit(X_train,Y_train)

#Exactitud, coeficientes e intercepto
print(logreg.score(X_train,Y_train))
print (logreg.intercept_, logreg.coef_)

#Predicciones
predict_y = logreg.predict(X_test)




#Exactitud (accuracy)
print ('Exactitud según scikit-learn: {0}'.format(logreg.score(X_test,Y_test)))

#Búsqueda en cuadrícula (GridSearch)
parameters = {'C':np.linspace(0.01,1,10)}
clf = GridSearchCV(LogisticRegression(solver='lbfgs'),parameters,cv=5)
clf.fit(X_train,Y_train)

#Mejor exactitud y mejores parámetros
print(clf.best_score_)
print(clf.best_params_)

#Exactitud después de GridSearch
print ('Exactitud después de GridSearch: {0}'.format(clf.score(X_test,Y_test)))

#Clasificación multiclase

#División en entrenamiento y prueba
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train.shape,X_test.shape)
print(Y_train.shape,Y_test.shape)

#Regresión logística
logreg_multi = LogisticRegression(solver='lbfgs',multi_class='ovr')
logreg_multi.fit(X_train,Y_train)

#Métricas
print(logreg_multi.score(X_train,Y_train))
print (logreg_multi.intercept_, logreg_multi.coef_)

#Predicciones
predict_y = logreg_multi.predict(X_test)
predict_y

#Exactitud (accuracy)
print ('Exactitud según scikit-learn: {0}'.format(logreg_multi.score(X_test,Y_test)))

#Matriz de confusión 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, predict_y)

#Búsqueda en cuadrícula (GridSearch)
parameters = {'C':np.linspace(0.01,1,10)}
clf_multi = GridSearchCV(LogisticRegression(solver='lbfgs',multi_class='ovr'),parameters,cv=5)
clf_multi.fit(X_train,Y_train)

#Mejor exactitud y mejores parámetros
print(clf_multi.best_score_)
print(clf_multi.best_params_)
print(clf_multi.best_estimator_)

#Predicciones
predict_y =clf_multi.predict(X_test)
predict_y

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, predict_y)

#Exactitud después de GridSearch
print ('Exactitud después de GridSearch: {0}'.format(clf_multi.score(X_test,Y_test)))

#Guardar y volver a cargar el modelo
import pickle

pickle.dump(clf_multi, open('model.pkl', 'wb'))

pickled_model = pickle.load(open('model.pkl', 'rb'))
pickled_model.predict(X_test)
