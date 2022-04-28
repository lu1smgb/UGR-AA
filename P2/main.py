# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Luis Miguel Guirado Bautista
"""
import numpy as np
import matplotlib.pyplot as plt
from sys import exit

# Fijamos la semilla
np.random.seed(1)

def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gauss(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out


def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b


# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente

# ***** Primera grafica ****************************************
x = simula_unif(50, 2, [-50,50])
#CODIGO DEL ESTUDIANTE
print("Distribución uniforme")
plt.scatter(x[:,0], x[:,1])
plt.title('Distribución uniforme de 50 puntos sobre [-50,50]')
plt.show()
# **************************************************************
input("\n--- Pulsar tecla para continuar ---\n")
# ***** Segunda grafica ****************************************
x = simula_gauss(50, 2, np.array([5,7]))
#CODIGO DEL ESTUDIANTE
print("Distribución Gaussiana")
plt.scatter(x[:,0], x[:,1])
plt.title('Distribución gaussiana de 50 puntos con $\sigma_x = 5$ y $\sigma_y = 7$')
plt.show()
# **************************************************************

# Comentado por que no se pide
"""a, b = simula_recta([-50,50])
print(a, b)
y = [a*x+b for x in np.linspace(-50,50,10)]
plt.plot(np.linspace(-50,50,10), y, 'C0')
plt.show()"""

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################


# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)

#CODIGO DEL ESTUDIANTE

print("Clasificacion sin ruido")
# ***** Apartado A (clasificacion sin ruido) *******************
# Generamos la muestra
x = simula_unif(100, 2, [-50,50])

# Coeficientes de la recta
a, b = simula_recta([-50,50])

# Etiquetas
y = np.array([f(xi, yi, a, b) for (xi, yi) in x], dtype=float)
plt.scatter(x[:,0], x[:,1], c=y)

# Puntos para representar la recta en el plot
x_ = np.linspace(-60,60,x.shape[0])
plt.plot(x_, a*x_ + b, 'C0')
plt.xlim(-55,55)
plt.ylim(-55,55)
plt.title('Clasificacion perfecta de 100 puntos según $f(x,y)=y-a \cdot x - b$')
plt.show()
# **************************************************************

# ! APUNTES PLA (BORRAR)
"""
    ||w||·||alpha||·cos ang.
    cos ang. determina signo
    
    PAGINA 12
    w^(t) son los pesos al final de una epoca
    Una epoca son todos los resultados del
    perceptron al haber pasado por todos los datos
    
    POCKET es PLA pero guardando solo lo
    que tiene minimo error (pag 22)
"""

input("\n--- Pulsar tecla para continuar ---\n")

# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido

#CODIGO DEL ESTUDIANTE
print("Clasificacion con ruido")
# ***** Apartado B (clasificacion con 10% de ruido en ambas etiquetas) ********
def generar_ruido(y, tasa_ruido):
    positivos = [i for i,_ in enumerate(y) if y[i]==1]
    negativos = [i for i,_ in enumerate(y) if y[i]==-1]

    ruido_pos = np.random.choice(positivos, 
                                 size=round(y[y==1].size*tasa_ruido),
                                 replace=False)

    ruido_neg = np.random.choice(negativos,
                                 size=round(y[y==-1].size*tasa_ruido),
                                 replace=False)

    ruido = np.concatenate((ruido_pos, ruido_neg))
    ruido = np.array(ruido, dtype=int)

    if ruido.size > 0:
        y[ruido] *= -1
# *****************************************************************************
tasa_ruido = 0.1

generar_ruido(y, tasa_ruido)

# Para el ejercicio 2
y_ruido = y.copy()

print(f"Tasa de ruido en ambas etiquetas: {round(tasa_ruido*100)} %")
print(f"Tasa de positivos {y[y==1].size} %\nPuntos de ruido positivos {round(y[y==1].size*tasa_ruido)}")
print(f"Tasa de negativos {y[y==-1].size} %\nPuntos de ruido negativos {round(y[y==-1].size*tasa_ruido)}")

# Mostramos la grafica
plt.scatter(x[:,0], x[:,1], c=y)
plt.plot(x_, a*x_ + b, 'C0')
plt.xlim(-55,55)
plt.ylim(-55,55)
plt.title(f'Clasificacion con {round(tasa_ruido*100)}% de ruido de 100 puntos según $f(x,y)=y-a \cdot x - b$')
plt.show()
# **************************************************************
input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta

# =============================================================================
# def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
#     #Preparar datos
#     min_xy = X.min(axis=0)
#     max_xy = X.max(axis=0)
#     border_xy = (max_xy-min_xy)*0.01
#     
#     #Generar grid de predicciones
#     xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
#                       min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
#     grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
#     pred_y = fz(grid)
#     # pred_y[(pred_y>-1) & (pred_y<1)]
#     pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
#     
#     #Plot
#     f, ax = plt.subplots(figsize=(8, 6))
#     contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
#     ax_c = f.colorbar(contour)
#     ax_c.set_label('$f(x, y)$')
#     ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
#     ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
#                 cmap="RdYlBu", edgecolor='white')
#     
#     XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
#     positions = np.vstack([XX.ravel(), YY.ravel()])
#     ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')
#     
#     ax.set(
#        xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
#        ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
#        xlabel=xaxis, ylabel=yaxis)
#     plt.title(title)
#     plt.show()
# =============================================================================
    
    
#CODIGO DEL ESTUDIANTE

def f1(x, y):
    return (x-10)**2 + (y-20)**2 - 400

def f2(x, y):
    return 0.5*(x+10)**2 + (y-20)**2 - 400

def f3(x, y):
    return 0.5*(x-10)**2 - (y+20)**2 - 400

def f4(x, y):
    return y - 20*(x**2) - 5*x + 3

print('Clasificacion sin ruido con funciones no lineales')
fig, axs = plt.subplots(2,2)
fig.set_size_inches((12,8))
X, Y = np.meshgrid(np.arange(-55, 55, 1), np.arange(-55, 55, 1))

y = np.array([signo(f1(xi, yi)) for (xi, yi) in x], dtype=float)
axs[0, 0].scatter(x[:, 0], x[:, 1], c=y)
axs[0, 0].contour(X, Y, f1(X, Y), 0)
axs[0, 0].set_title(
	f'Clasificacion perfecta con $f_1$')

y = np.array([signo(f2(xi, yi)) for (xi, yi) in x], dtype=float)
axs[0, 1].scatter(x[:, 0], x[:, 1], c=y)
axs[0, 1].contour(X, Y, f2(X, Y), 0)
axs[0, 1].set_title(
	f'Clasificacion perfecta con $f_2$')

y = np.array([signo(f3(xi, yi)) for (xi, yi) in x], dtype=float)
axs[1, 0].scatter(x[:, 0], x[:, 1], c=y)
axs[1, 0].contour(X, Y, f3(X, Y), 0)
axs[1, 0].set_title(
    f'Clasificacion perfecta con $f_3$')

y = np.array([signo(f4(xi, yi)) for (xi, yi) in x], dtype=float)
axs[1, 1].scatter(x[:, 0], x[:, 1], c=y)
axs[1, 1].contour(X, Y, f4(X, Y), 0)
axs[1, 1].set_title(
	f'Clasificacion perfecta con $f_4$')

plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

print('Clasificacion con ruido con funciones no lineales')
fig, axs = plt.subplots(2,2)
fig.set_size_inches((12,8))

y = np.array([signo(f1(xi, yi)) for (xi, yi) in x], dtype=float)
generar_ruido(y,tasa_ruido)
axs[0, 0].scatter(x[:, 0], x[:, 1], c=y)
axs[0, 0].contour(X, Y, f1(X, Y), 0)
axs[0, 0].set_title(
	f'Clasificacion ruidosa con $f_1$')

y = np.array([signo(f2(xi, yi)) for (xi, yi) in x], dtype=float)
generar_ruido(y,tasa_ruido)
axs[0, 1].scatter(x[:, 0], x[:, 1], c=y)
axs[0, 1].contour(X, Y, f2(X, Y), 0)
axs[0, 1].set_title(
	f'Clasificacion ruidosa con $f_2$')

y = np.array([signo(f3(xi, yi)) for (xi, yi) in x], dtype=float)
generar_ruido(y,tasa_ruido)
axs[1, 0].scatter(x[:, 0], x[:, 1], c=y)
axs[1, 0].contour(X, Y, f3(X, Y), 0)
axs[1, 0].set_title(
    f'Clasificacion ruidosa con $f_3$')

y = np.array([signo(f4(xi, yi)) for (xi, yi) in x], dtype=float)
generar_ruido(y,tasa_ruido)
axs[1, 1].scatter(x[:, 0], x[:, 1], c=y)
axs[1, 1].contour(X, Y, f4(X, Y), 0)
axs[1, 1].set_title(
	f'Clasificacion ruidosa con $f_4$')

plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 2.1: ALGORITMO PERCEPTRON

def ajusta_PLA(datos: np.array, label: np.array, max_iter, vini: np.ndarray):
    #CODIGO DEL ESTUDIANTE
    assert vini.shape in [(3,),(1,3)]
    w = vini
    iters = 0
    datos_ = np.insert(datos, 0, np.ones(x.shape[0]), axis=1)
    cambios = True # Para entrar en el bucle
    while cambios and iters < max_iter:
        cambios = False
        for idx,xi in enumerate(datos_):
            iters += 1
            if signo(w.dot(xi)) != label[idx]:
                w = w + xi*label[idx]
                cambios = True
    return w, iters

#CODIGO DEL ESTUDIANTE
MAX_ITERS = 50_000
# Recuperamos las etiquetas del ejercicio 1.2.a
y = np.array([f(xi, yi, a, b) for (xi, yi) in x], dtype=float)
print('Algoritmo PLA con vector 0')
pesos, iters = ajusta_PLA(x, y, MAX_ITERS, np.zeros(3, dtype=int))
print(pesos, iters, '\n')

# Random initializations
print('Algoritmo PLA con vectores aleatorios')
iterations = []
for i in range(0,10):
    #CODIGO DEL ESTUDIANTE
    wini = np.random.sample(3)
    pesos, iters = ajusta_PLA(x, y, MAX_ITERS, wini)
    print(f'\t--- Iteración {i+1} ---')
    print(f'Pesos iniciales: {wini}')
    print(f'Pesos finales: {pesos}')
    print(f'Iteraciones empleadas: {iters}')
    iterations.append(iters)
    
print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

input("\n--- Pulsar tecla para continuar ---\n")

# Ahora con los datos del ejercicio 1.2.b

#CODIGO DEL ESTUDIANTE
print('\t***** Ahora con ruido *****')
print('Algoritmo PLA con vector 0')
pesos, iters = ajusta_PLA(x, y_ruido, 500_000, np.zeros(3, dtype=int))
print(pesos, iters, '\n')

# Random initializations
print('Algoritmo PLA con vectores aleatorios')
iterations = []
for i in range(0,10):
    #CODIGO DEL ESTUDIANTE
    wini = np.random.sample(3)
    pesos, iters = ajusta_PLA(x, y_ruido, MAX_ITERS, wini)
    print(f'Iteración {i+1}')
    print(f'Pesos iniciales: {wini}')
    print(f'Pesos finales: {pesos}')
    print(f'Iteraciones empleadas: {iters}')
    iterations.append(iters)
    
print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

exit()
def sgdRL():
    #CODIGO DEL ESTUDIANTE

    pass



#CODIGO DEL ESTUDIANTE



input("\n--- Pulsar tecla para continuar ---\n")
    


# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).


#CODIGO DEL ESTUDIANTE




input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################
#BONUS: Clasificación de Dígitos


# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])


#mostramos los datos
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#LINEAR REGRESSION FOR CLASSIFICATION 

#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")



#POCKET ALGORITHM
  
#CODIGO DEL ESTUDIANTE




input("\n--- Pulsar tecla para continuar ---\n")


#COTA SOBRE EL ERROR

#CODIGO DEL ESTUDIANTE
