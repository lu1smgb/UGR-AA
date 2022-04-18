# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: Luis Miguel Guirado Bautista
"""

import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from inspect import getfullargspec

import pandas

rnd.seed(1)

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1\n')

# ************************************************************************

# Funcion E del apartado 2
def E(u,v):
    return (u*v*np.e**(-u**2-v**2))**2

# Derivada parcial de E con respecto a u
def dEu(u,v):
    return (2*u*v**2)*np.e**(-2*u**2-2*v**2)*(1-2*u**2)
    
# Derivada parcial de E con respecto a v
def dEv(u,v):
    return (2*v*u**2)*np.e**(-2*u**2-2*v**2)*(1-2*v**2)
    # Calcular a mano

#Gradiente de E
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])

# ************************************************************************

# Funcion f del apartado 3
def f(x,y):
    return x**2 + 2*y**2 + 2*np.sin(2*np.pi*x)*np.sin(np.pi*y)

# Derivada parcial de f con respecto a x
def dfx(x,y):
    return 2*x + 4*np.pi*np.sin(np.pi*y)*np.cos(2*np.pi*x)

# Derivada parcial de f con respecto a y
def dfy(x,y):
    return 4*y + 2*np.pi*np.sin(2*np.pi*x)*np.cos(np.pi*y)

# Gradiente de f
def gradf(x,y):
    return np.array([dfx(x,y), dfy(x,y)])

# *************************************************************************

# Algoritmo de gradiente descendente para el ejercicio de búsqueda iterativa de óptimos
def gradient_descent(w_ini, lr, grad_fun, fun, epsilon, max_iters, display=False):
    """

    Algoritmo de gradiente descendente

    Parámetros:
    
    w_ini: (float, float)
        Punto inicial del que parte el algoritmo

    lr: float
        Tasa de aprendizaje del algoritmo

    grad_fun: funcion
        Gradiente de la función fun

    fun: funcion
        Funcion a minimizar

    max_iters: int
        Iteraciones máximas

    display: bool = False
        Decidir si imprime por pantalla el resultado de cada iteración

    """
    iterations = 0
    w = w_ini
    ws = [w]
    if display: print(f'[{iterations}] ({w[0]}, {w[1]}) = {fun(w[0],w[1])}')
    while fun(w[0],w[1]) > epsilon and iterations < max_iters:
        iterations += 1
        w = w - lr * grad_fun(w[0],w[1])
        ws.append(w)
        if display: print(f'[{iterations}] ({w[0]}, {w[1]}) = {fun(w[0],w[1])}')

    return w, iterations, ws


def display_figure(rng_val, fun, ws, colormap, title_fig):
    '''
    Esta función muestra una figura 3D con la función a optimizar junto con el 
    óptimo encontrado y la ruta seguida durante la optimización. Esta función, al igual
    que las otras incluidas en este documento, sirven solamente como referencia y
    apoyo a los estudiantes. No es obligatorio emplearlas, y pueden ser modificadas
    como se prefiera. 
        rng_val: rango de valores a muestrear en np.linspace()
        fun: función a optimizar y mostrar
        ws: conjunto de pesos (pares de valores [x,y] que va recorriendo el optimizador
                            en su búsqueda iterativa del óptimo)
        colormap: mapa de color empleado en la visualización
        title_fig: título superior de la figura
        
    Ejemplo de uso: display_figure(2, E, ws, 'plasma','Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')
    '''
    from mpl_toolkits.mplot3d import Axes3D
    x = np.linspace(-rng_val, rng_val, 50)
    y = np.linspace(-rng_val, rng_val, 50)
    X, Y = np.meshgrid(x, y)
    Z = fun(X, Y) 
    fig = plt.figure()
    ax = Axes3D(fig,auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                            cstride=1, cmap=colormap, alpha=.6)
    if len(ws)>0:
        ws = np.asarray(ws)
        min_point = np.array([ws[-1,0],ws[-1,1]])
        min_point_ = min_point[:, np.newaxis]
        # ax.plot(ws[:-1,0], ws[:-1,1], fun(ws[:-1,0], ws[:-1,1]), 'r*', markersize=5)

        ax.plot(ws[:,0], ws[:,1], fun(ws[:,0], ws[:,1]), 'r-', markersize=2)

        ax.plot(min_point_[0], min_point_[1], fun(min_point_[0], min_point_[1]), 'r*', markersize=10)
    if len(title_fig)>0:
        fig.suptitle(title_fig, fontsize=16)
    # https://www.tutorialspoint.com/How-to-get-a-list-of-parameter-names-inside-Python-function#:~:text=To%20extract%20the%20number%20and,the%20functions%20aMethod%20and%20foo.
    fun_params = getfullargspec(fun).args
    ax.set_xlabel(f'{fun_params[0]}')
    ax.set_ylabel(f'{fun_params[1]}')
    # Uso de __name__: https://stackoverflow.com/questions/251464/how-to-get-a-function-name-as-a-string
    ax.set_zlabel(f'{fun.__name__}({fun_params[0]},{fun_params[1]})')
    plt.show()

def ej12():
    eta = 0.1 
    maxIter = 10e10
    error2get = 1e-8
    initial_point = np.array([0.5,-0.5])
    w, it, ws1 = gradient_descent(initial_point, eta, gradE, E, error2get, maxIter, display=False)
    print ('Numero de iteraciones: ', it)
    print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
    print('Valor alcanzado: ', f(w[0],w[1]))
    display_figure(1.5*np.abs(initial_point[0]), E, ws1, 'viridis', '')
    input("\n--- Pulsar tecla para continuar ---\n")

def ej13a(eta=0.01, initial_point=np.array([-1.0,1.0]), display = True, debug = True):
    maxIter = 50
    error2get = -1000000
    w, it, ws = gradient_descent(initial_point, eta, gradf, f, error2get, maxIter, display)
    if debug:
        print ('Numero de iteraciones: ', it)
        print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
        print('Valor alcanzado: ', f(w[0],w[1]))
    if display: display_figure(3*np.abs(initial_point[0]), f, ws, 'viridis', '')
    if debug: input("\n--- Pulsar tecla para continuar ---\n")
    return w, f(w[0],w[1])# Los necesitaremos en el main

def ej13b():
    print('Tabla correspondiente al ejercicio 1.3, apartado B')
    lrs = [0.1, 0.01]
    puntos = [[-0.5, 0.5], [1, 1], [2.1, -2.1], [-3, 3], [-2, 2]]
    tabla_dict = {
        '0.1': [],
        '0.01': []
    }
    for lr in lrs:
        for punto in puntos:
            final, valor = ej13a(lr, np.asarray(punto), display=False, debug=False)
            final = np.round(final,3)
            valor = np.round(valor,3)
            tabla_dict[str(lr)].append((final, valor))

    tabla = pandas.DataFrame(tabla_dict, index=[str(punto) for punto in puntos])
    print(tabla)
    input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################

# *** Funciones del fichero 'funciones_utils.py'
def simula_unif(N=2, dims=2, size=(0, 1)):
    m = np.random.uniform(low=size[0], 
                          high=size[1], 
                          size=(N, dims))
    
    return m


def label_data(x1, x2):
    y = np.sign((x1-0.2)**2 + x2**2 - 0.6)
    idx = np.random.choice(range(y.shape[0]), size=(int(y.shape[0]*0.1)), replace=True)
    y[idx] *= -1
    
    return y

print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 1\n')

label5 = 1
label1 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Funcion para calcular el error
# x -> parametros de la funcion predictora (size=3)
# y -> valor objetivo
# w -> modelo de la funcion predictora (size=3)
def Err(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    suma = 0
    errores = 0
    n = len(x)
    for i in range(n):
        # Mult. matriz 3x1 y 1x3 -> Un unico numero
        hx = sign(x[i].dot(w))
        if hx != y[i]:
            errores += 1
        suma += (hx - y[i])**2
    return suma/n, errores

# Devuelve el vector gradiente
def dErr(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    suma = 0
    m = len(x)
    for i in range(m):
        hx = sign(x[i].dot(w))
        suma += x[i]*(hx - y[i])
    return 2*suma/m

# Gradiente Descendente Estocastico
def sgd(eta: float, x: np.ndarray, y: np.ndarray):

    iters = 0
    max_iters = 400
    w = np.zeros(x.shape[1], dtype=float)

    # Encontrar un tamaño de subconjunto adecuado
    bsize = 2
    while len(x) % bsize != 0 and bsize < len(x)//2:
        bsize += 1
    batches = len(x) // bsize
    # print(f'Tamaño de los subconjuntos apropiado: {bsize}')
    # print(f'Número de subconjuntos: {batches}')

    conjunto = x.copy()
    aux = np.reshape(y, (len(y),1))
    conjunto = np.append(conjunto, aux, axis=1)
    # conjunto queda [ ...
    #                 [x0,x1,x2,y],
    #                  ...         ]
    while True:
        # Creamos los subconjuntos desordenados previamente
        # conjunto seran los datos junto a su objetivo
        rnd.shuffle(conjunto)
        minibatches = np.split(conjunto, batches)
        for minib in minibatches:
            iters += 1
            w = w - eta*dErr(minib[:,0:-1], minib[:,-1], w)
            # print(f'{iters} -> {w} \n\t {float(Err(x, y, w)):.3f} \n\t {dErr(minib[:,0:-1], minib[:,-1], w)}\n')
            if iters >= max_iters: return w



# Pseudoinversa	
# https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/#:~:text=7.%C2%A0%C2%A08.%C2%A0%C2%A09.%5D%5D-,SVD%20for%20Pseudoinverse,-The%20pseudoinverse%20is
def pseudoinverse(x: np.ndarray, y: np.ndarray):

    w = np.zeros((x.shape[1],), dtype=float)
    
    u, d, vt = np.linalg.svd(x)
    
    # Muy importante esta transformacion
    # d es un array unidimensional de n longitud
    # diag lo pasa a una matriz diagonal de nxn
    D = np.diag(d)
    
    # Es IMPRESCINDIBLE usar la función dot.
    # dot realiza el producto matricial de dos matrices
    # Esto no pasa con el operador *
    xtx = np.transpose(vt).dot(D).dot(D).dot(vt)
    xtx_1 = np.linalg.inv(xtx)
    pseudo = xtx_1.dot(np.transpose(x))
    
    w = np.dot(pseudo, y)

    return w

def ej21():
    # Lectura de los datos de entrenamiento
    x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
    # Lectura de los datos para el test
    x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')
    
    w = sgd(0.2, x, y)
    print ('Bondad del resultado para grad. descendente estocastico:\n')
    print(f'Modelo lineal obtenido: {w}')
    print ("Ein: ", Err(x, y, w))
    print ("Eout: ", Err(x_test, y_test, w))

    # GRAFICO PARA TRAINING
    x1 = []
    x2 = []
    for ele in range(len(y)):
        if y[ele] == 1: x1.append(x[ele])
        else: x2.append(x[ele])
    x1 = np.array(x1)
    x2 = np.array(x2)

    # Geométricamente hablando
    # Es importante que el modelo ahora deba representarse en el plano X,Y
    # Y no en el plano X_1, X_2
    modelo = [w[0]+((w[1]+w[2])*x_) for x_ in np.linspace(0,0.6,6)]

    plt.title("Números manuscritos (training)")
    plt.xlabel("Nivel medio de gris")
    plt.ylabel("Simetría")
    plt.plot(x1[:,1], x1[:,2], 'ro', label='5')
    plt.plot(x2[:,1], x2[:,2], 'bo', label='1')
    plt.plot(np.linspace(0,0.6,6), modelo, 'g-', label='w')
    plt.legend()
    plt.xlim(0,0.6)
    plt.ylim(-8,0)
    print("Mostrando training...")
    plt.show()
    
    input("\n--- Pulsar tecla para continuar ---\n")
    
    # GRAFICO PARA TEST
    x1_t = []
    x2_t = []
    for ele in range(len(y_test)):
        if y_test[ele] == 1: x1_t.append(x_test[ele])
        else: x2_t.append(x_test[ele])
    x1_t = np.array(x1_t)
    x2_t = np.array(x2_t)
    
    plt.title("Números manuscritos (test)")
    plt.xlabel("Nivel medio de gris")
    plt.ylabel("Simetría")
    plt.plot(x1_t[:,1], x1_t[:,2], 'ro', label='5')
    plt.plot(x2_t[:,1], x2_t[:,2], 'bo', label='1')
    plt.plot(np.linspace(0,0.6,6), modelo, 'g-', label='w')
    plt.legend()
    plt.xlim(0,0.6)
    plt.ylim(-8,0)
    print("Mostrando test...")
    plt.show()
    
    input("\n--- Pulsar tecla para continuar ---\n")
    
    # **********************************************************
    
    w = pseudoinverse(x,y)
    print ('Bondad del resultado para pseudoinversa:\n')
    print(f'Modelo lineal obtenido: {w}')
    print ("Ein: ", Err(x, y, w))
    print ("Eout: ", Err(x_test, y_test, w))

    # GRAFICO PARA TRAINING
    x1 = []
    x2 = []
    for ele in range(len(y)):
        if y[ele] == 1: x1.append(x[ele])
        else: x2.append(x[ele])
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    modelo = [w[0]+((w[1]+w[2])*x_) for x_ in np.linspace(0,0.6,6)]

    plt.title("Números manuscritos (training)")
    plt.xlabel("Nivel medio de gris")
    plt.ylabel("Simetría")
    plt.plot(x1[:,1], x1[:,2], 'ro', label='5')
    plt.plot(x2[:,1], x2[:,2], 'bo', label='1')
    plt.plot(np.linspace(0,0.6,6), modelo, 'g-')
    plt.xlim(0,0.6)
    plt.ylim(-8,0)
    plt.legend()
    print("Mostrando training...")
    plt.show()
    
    input("\n--- Pulsar tecla para continuar ---\n")
    
    # GRAFICO PARA TEST
    x1_t = []
    x2_t = []
    for ele in range(len(y_test)):
        if y_test[ele] == 1: x1_t.append(x_test[ele])
        else: x2_t.append(x_test[ele])
    x1_t = np.array(x1_t)
    x2_t = np.array(x2_t)
    
    plt.title("Números manuscritos (test)")
    plt.xlabel("Nivel medio de gris")
    plt.ylabel("Simetría")
    plt.plot(x1_t[:,1], x1_t[:,2], 'ro', label='5')
    plt.plot(x2_t[:,1], x2_t[:,2], 'bo', label='1')
    plt.plot(np.linspace(0,0.6,6), modelo, 'g-', label='w')
    plt.legend()
    plt.xlim(0,0.6)
    plt.ylim(-8,0)
    print("Mostrando test...")
    plt.show()

    input("\n--- Pulsar tecla para continuar ---\n")

print('Ejercicio 2\n') #**********************************************************+
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

def sign(x):
	if x >= 0:
		return 1
	return -1

def f2(x1, x2):
	return sign((x1-0.2)**2+x2**2-0.6)

def gen_muestra_lineal(N: int=1000):

    # *a) #
    data = simula_unif(N,2,1)

    # *b) #
    # Generamos las etiquetas para cada punto
    labels = np.asarray([f2(punto[0],punto[1]) for punto in data])
    # Cambiamos el 10% de los puntos
    for index in np.random.randint(0,N,N//10):
        labels[index] = -labels[index]

    # *c) #
    # Modificamos los datos para que cada punto tenga el formato (1,x1,x2)
    data = np.insert(data, 0, 1, axis=1)

    return data, labels

def ej22():

    N = 1000
    # Generamos una muestra: datos y etiquetas / X e y
    data, labels = gen_muestra_lineal(N)
    
    # Mostramos el gráfico
    plt.scatter(data[:,1], data[:,2], c=labels)

    # Estimacion del modelo
    w = sgd(0.1, data, labels)
    print ('Bondad del resultado para grad. descendente estocastico:\n')
    print(f'Modelo lineal obtenido: {w}')
    print ("Ein: ", Err(data, labels, w))

    modelo = [w[0]+np.sum(w[1:]*x_) for x_ in np.linspace(-1,1,20)]
    plt.plot(np.linspace(-1,1,20), modelo, 'g-')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.show()

    input("\n--- Pulsar tecla para continuar ---\n")

    # *d) #
    # Ahora realizamos lo anterior 1000 veces y mostramos los errores medios
    print(f'Generando 1000 muestras para regresión lineal, espere por favor...')
    ein_media = eout_media = ein_errores = eout_errores = 0
    for _ in range(N):
        # Calculamos Ein
        data, labels = gen_muestra_lineal(N)
        w = sgd(0.1, data, labels)
        ein, err = Err(data, labels, w)
        ein_media += ein
        ein_errores += err

        # Calculamos Eout
        data, labels = gen_muestra_lineal(N)
        eout, err = Err(data, labels, w)
        eout_media += ein
        eout_errores += err
    ein_media /= N
    eout_media /= N
    ein_errores /= N*10
    eout_errores /= N*10
    print('Error medio en 1000 muestras con modelo lineal: ')
    print(f'Ein = {ein_media, ein_errores}')
    print(f'Eout = {eout_media, eout_errores}')

    # *e) #
    ein_media = 0
    eout_media = 0
    ein_errores = 0
    eout_errores = 0
    def nolinear_reg_function(x, y, w):
        return w[0] + w[1]*x + w[2]*y + w[3]*x*y + w[4]*x**2 + w[5]*y**2
    
    print(f'Generando 1000 muestras para regresión no lineal, espere por favor...')
    for _ in range(N):
        data, labels = gen_muestra_lineal(N)
        data_test, labels_test = gen_muestra_lineal(N)
        # Modificamos cada x para que tengan formato no linear
        # x1x2
        data = np.append(data, (data[:,1]*data[:,2]).reshape((data.shape[0],1)), axis=1)
        data_test = np.append(data_test, (data_test[:,1]*data_test[:,2]).reshape((data_test.shape[0],1)), axis=1)
        # x1^2
        data = np.append(data, (data[:,1]**2).reshape((data.shape[0],1)), axis=1)
        data_test = np.append(data_test, (data_test[:,1]**2).reshape((data_test.shape[0],1)), axis=1)
        # x2^2
        data = np.append(data, (data[:,2]**2).reshape((data.shape[0],1)), axis=1)
        data_test = np.append(data_test, (data_test[:,2]**2).reshape((data_test.shape[0],1)), axis=1)

        # Y ahora generamos el modelo no lineal
        w = sgd(0.1, data, labels)
        ein, err = Err(data, labels, w)
        ein_media += ein
        ein_errores += err
        eout, err = Err(data_test, labels_test, w)
        eout_media += eout
        eout_errores += err

    ein_media /= N
    eout_media /= N
    ein_errores /= N*10
    eout_errores /= N*10
    print('Error medio en 1000 muestras con modelo no lineal: ')
    print(f'Ein = {ein_media,ein_errores}')
    print(f'Eout = {eout_media,eout_errores}')
    # https://matplotlib.org/stable/gallery/images_contours_and_fields/contour_demo.html
    x = np.arange(-1,1,0.1)
    y = np.arange(-1,1,0.1)
    X, Y = np.meshgrid(x, y)
    Z = nolinear_reg_function(X, Y, w)
    plt.scatter(data[:,1], data[:,2], c=labels)
    plt.contour(X, Y, Z, levels=1)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.show()

def main():
    
    # # ***  EJERCICIO 1.2  *** #
    """ ej12()

    # # *** EJERCICIO 1.3 A *** #
    ej13a(0.01)
    ej13a(0.1)

    # # *** EJERCICIO 1.3 B *** #
    ej13b()

    # ***  EJERCICIO 2.1    *** #
    ej21() """

    # ***  EJERCICIO 2.2    *** #
    ej22()

    print('*** Final del programa ***')

    exit()

if __name__ == "__main__":
    main()