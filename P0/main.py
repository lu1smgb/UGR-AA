# Práctica 0
# Luis Miguel Guirado Bautista

# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris as descargar_datos
import matplotlib.pyplot as plt
from matplotlib import cm # Colormaps
import numpy as np 
import cmath # Número e

# ******************************************************************************************

def ej1():

   # Cargamos los datos en nuestro programa
   # Tupla con: (matriz 2D con los datos, array con la clase de cada flor)
   datos = descargar_datos(return_X_y=True)

   # Longitud del sepalo de todas las clases (coordenadas X)
   sepal = tuple(x[0] for x in datos[0])

   # Longitud del petalo de todas las clases (coordenadas Y)
   petal = tuple(x[2] for x in datos[0])

   # Ya no nos hará falta (hacer esto no es del todo necesario)
   del datos

   # Tamaño de los puntos del gráfico
   psize = 30

   # Construcción de la gráfica
   # Representamos los grupos de datos uno por uno
   plt.title('Ejercicio 1')
   plt.scatter(sepal[:50], petal[:50], psize, '#ef0000', label='setosa')
   plt.scatter(sepal[50:100], petal[50:100], psize, '#00af00', label='versicolor')
   plt.scatter(sepal[100:], petal[100:], psize, '#0000ef', label='virginica')
   plt.xlabel('sepal lenght (cm)')
   plt.ylabel('petal lenght (cm)')
   plt.legend(loc='upper left')
   plt.show()

# ******************************************************************************************

def ej2():

   # Inicializamos los conjuntos de datos para el ejercicio
   # Solo necesitamos las clases
   datos = descargar_datos(return_X_y=True)
   training = list(datos[1].copy())
   test = []
   del datos
   np.random.shuffle(training)

   # Selección aleatoria de las muestras
   # Hasta que alcancemos el 20% de las muestras totales en test...
   while len(test) < 30:
      indice = np.random.randint(len(training)) # Indice aleatorio
      candidato = training[indice]
      # Evitamos tener más del 20% de cada clase
      if test.count(candidato) >= 10:
         continue # Escogemos otro elemento
      else:
         test.append(candidato)
         training.pop(indice)

   # Imprimimos el resultado
   print('*** Ejercicio 2 ***')
   print('--- Clase setosa ---')
   print(f'Ejemplos training: {training.count(0)}')
   print(f'Ejemplos test: {test.count(0)}\n')
   print('--- Clase versicolor ---')
   print(f'Ejemplos training: {training.count(1)}')
   print(f'Ejemplos test: {test.count(1)}\n')
   print('--- Clase virginica ---')
   print(f'Ejemplos training: {training.count(2)}')
   print(f'Ejemplos test: {test.count(2)}\n')
   print(f'Clase de los ejemplos de entrenamiento:\n{training}\n')
   print(f'Clase de los ejemplos de test:\n{test}\n')
   input(" --- Presione una tecla para continuar --- ")

# ******************************************************************************************

def ej3():
   # Valores de X
   valores = list(np.linspace(0,4*np.pi,100))

   # Valores de Fn(X) (nuestras funciones)
   f1 = [(10**-5)*cmath.sinh(x) for x in valores]
   f2 = [cmath.cos(x) for x in valores]
   f3 = [cmath.tanh(2*cmath.sin(x)-4*cmath.cos(x)) for x in valores]

   # Mostramos las funciones en la gráfica
   plt.title('Ejercicio 3')
   
   # ComplexWarning: Casting complex values to real discards the imaginary part
   # Se puede omitir con estas instrucciones
   # Referencia: https://stackoverflow.com/questions/41001533/how-to-ignore-python-warnings-for-complex-numbers
   import warnings
   warnings.filterwarnings('ignore')

   plt.plot(valores, f1, 'g--', label='y = 1e-5*sinh(x)')
   plt.plot(valores, f2, 'k--', label='y = cos(x)')
   plt.plot(valores, f3, 'r--', label='y = tanh(2*sin(x)-4*cos(x))')
   plt.legend(loc='upper left')
   plt.show()

# ******************************************************************************************

def ej4():

   # Funciones matematicas
   fn1 = lambda x,y: 1-abs(x+y)-abs(y-x)
   fn2 = lambda x,y: x*y*cmath.e**(-x**2-y**2)

   # Valores de X y de Y en forma de matrices de coordenadas
   fn1x = np.array(np.linspace(-6,6,30))
   fn2x = np.array(np.linspace(-2.5,2.5,100))
   fn1x, fn1y = np.meshgrid(fn1x, fn1x)
   fn2x, fn2y = np.meshgrid(fn2x, fn2x)

   # Valores de la función (Z)
   fn1z = fn1(fn1x,fn1y)
   fn2z = fn2(fn2x,fn2y)

   # Creacion y visualizacion de los graficos

   # Para que en la misma figura puedan dibujarse dos gráficos
   fig = plt.figure(figsize=plt.figaspect(0.5))

   # Inicializamos los espacios donde visualizar los gráficos
   ax1 = fig.add_subplot(1,2,1, projection='3d', title='Pirámide')
   ax2 = fig.add_subplot(1,2,2, projection='3d', title=r'$x·y·e^{(-x^2-y^2)}$')

   # Y los dibujamos según las funciones
   ax1.plot_surface(fn1x, fn1y, fn1z, cmap=cm.coolwarm, antialiased=True)
   ax2.plot_surface(fn2x, fn2y, fn2z, cmap=cm.viridis, antialiased=True)

   plt.show()

# ******************************************************************************************

def main():
   ej1()
   ej2()
   ej3()
   ej4()

if __name__ == "__main__":
   main()