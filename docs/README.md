PONER INDEX


# Teoria

## Que es la Inteligencia Artificial

Para poder explicar cualquier cosa sobre deep learning y por que es tan importante, se ha de partir sobre la base de que es la Inteligencia arn de diversos algoritmos, se analiza el conjunto de datos. para que al final si se ingresa un dato similar (No igual) a estos, el programa con el aprendizaje previo predice que este dato pertenece a un conjunto que ya conoce.

De forma mas directa y a modo de ejemplo, un vehiculo autonomo es conducido por una persona durante mucho tiempo en muchas carreteras distintas, luego del proceso de aprendizaje si el auto (sin piloto), se enfrenta a una carretera de caracterizticas similares a las anteriores (una curva por ejemplo), va a poder ejecutar la accion requerida (virar si es una curva).

## DEEP LEARNING 

### Que es el Deep Learning

Deep learning o *Aprendizaje profundo* son conjuntos de algoritmos basados en **redes neuronales artificiales**, en palabras comunes es una tecnica del area del aprendizaje de maquina, que trata de simular el aprendizaje humano, esto es aprender con el ejemplo. 
La tecnica de Deep Learning es la responsable de los vehiculos autonomos, del reconocimiento facial y el control por voz en dispositivos tecnologicos. Haciendo que el publico general se interese cada vez mas en este fascinante mundo de la Inteligencia Artificial.

Para implementar la tecnica de Deep Learning primero hemos de conocer sus elementos, partiendo por su elemento mas basico. la Neurona

#### Neurona
La neurona artificial (Conocida como *Neurona de McCulloch-Pitts*) es una unidad de calculo que trata de imitar el comportamiento de una neurona natural. basicamente la neurona realiza una operacion de suma ponderada de sus entradas. y luego se aplica una funcion no lineal dando asi un resultado llamado output. (para motivos pedagogicos se puede interpretar una neurona como una funcion de n entradas que las multiplica por *pesos* y luego a su suma aplica una funcion no lineal llamada *funcion de activacion*)

El diseño de una neurona se puede ver:

tificial. la Inteligencia Artificial (tambien llamada inteligencia de computadoras ó IA) es inteligencia expresada por maquinas, mas formalmente se define como:
*"La habilidad de un sistema (computador) para interpretar correctamente datos externos, aprender de estos, y usar el aprendizaje para conseguir objetivos y tareas especificas a travez de una adaptacion flexible"*
En otras palabras la **IA** se refiere a algoritmos que permiten a un computador alimentarse de datos y obtener patrones sobre estos, sin estos ser programados en la maquina.

Una IA tipica analiza su entorno y realiza cambios para obtener un mejor resultado. para esto posee una *Utility Function* o tambien llamada variable objetivo, la cual puede ser explicitamente programada en la maquina o tambien inducida a travez de los datos. Luego a travez de la utilizacio
![Neurona de McCulloch-Pitts](https://upload.wikimedia.org/wikipedia/commons/c/c8/Mccullochpitts.png)

y la operacion matematica se veria como

*f*(*x*<sub>1</sub>, *x*<sub>2</sub>, …, *x*<sub>*n*</sub>) = *s*(*c*<sub>1</sub>*x*<sub>1</sub> + *c*<sub>2</sub>*x*<sub>2</sub> + … + *c*<sub>*n*</sub>*x*<sub>*n*</sub> − *d*)

Con:
* f la *funcion* que representa el output de la neurona
* \[x_i\] las entradas de la neurona
* \[c_i\] los pesos de las entradas
* s la funcion de activacion
* d el umbral de la operacion

Estas operaciones se pueden representar facilmente con matrizes y asi facilitar la implementacion de estas en programas de computadoras

#### Red Neuronal
Una red Neuronal son conjuntos de capas (*layers*) de neuronas artificiales conectadas entre si para transmitir señales. la señal de input se transmite sucesivamente aplicando la operacion correspondiente y luego de un numero de ciclos se genera una salida. Estas señales se modifican a travez de pesos, los cuales son variables y pueden aumentar o disminuir según nescesidad y asi afectar el estado de activacion de las neuronas adyacentes, de igual forma pueden existir funciones que imponen condiciones para que dicha señal pueda transmitirse mas alla. esta funcion se llama funcion de activacion.

La red neuronal y sus pesos no son programados de forma explicita, sino que con el objetivo de reducir su funcion perdida (*Loss function*), varian sus pesos y estructura para asi obtener un resultado optimo. para realizar el cambio de los pesos se usa el metodo de propagacion hacia atras o *backpropagation*

##### Loss Function
La funcion de perdida, o funcion de costo es una funcion que mapea un evento a un numero real que representa el costo o *perdida* de la operacion anterior, una funcion de perdida comun es la de perdida cuadratica que es basicamente la diferencia entre el valor obtenido menos el esperado al cuadrado. existen muchas funciones de perdida y se ha de ver según el problema cual sera usada para el modelo computacional.
##### Backpropagation
Backpropagation (Propagacion inversa) es un algoritmo cuyo objetivo es minimizar la funcion de perdida. esto se hace a travez del gradiente de la funcion de costo respecto a cada peso en la red.     

![Backpropagation](https://www.guru99.com/images/1/030819_0937_BackPropaga1.png)

basicamente el gradiente indica como cambia la funcion costo al modificar un peso especifico. Modificando los pesos y analizando con backpropagation podemos obtener un punto *optimo* de nuestra red neuronal donde la funcion loss es pequeña y asi obteniendo resultados mas exactos al momento de utilizar el modelo construido.

##### Hidden Layers
Las *hidden layers* son las capas de neuronas que no son ni el input ni el output. Estas son las capas donde los pesos son modificados por el algoritmo de backpropagation, se llaman hidden layers debido a que no sabemos como seran las relaciones entre las neuronas ni sus pesos de antemano.

##### Funciones de activacion
La funcion de activacion de un nodo de una red neuronal define la salida que tendra sobre un set de entradas, existen diversas funciones de activacion, tales como la identidad que deja la salida identica a la entrada, *binary pass* que da un 1 si la entrada es positiva y 0 en otro caso. o una de las mas usadadas ReLU (*Unidad lineal rectificada*) que entrega el maximo entre (0.000...1) y el mismo valor. esta funcion fue y continua siendo usada debido a que esta hecha para solucionar los errores propios de realizar operaciones matematicas complejas en un computador. Debido a sus problemas debidos a su vaga definicion actualmente existen otras funciones tales como la sigmoide que es mas precisa aunque requiere mayor procesamiento computacional.


#### Red Neuronal Convolusional (*CNN*)

En este Workshop además de la red neuronal, utilizaremos redes convolusionales. las cuales estan enfocadas en imitar el comportamiento de la vision de un cerebro biologico, su principal uso son para Vision artificial y generacion de imagenes. 
Las CNN's consisten en multiples capas de filtros convolusionales para asi ir reduciendo el numero de informacion menos util para el modelo. generalmente luego de aplicar el filtro, se aplican funciones para obtener las features de mayor relevancia de los datos.
Luego de procesar por un numero determinado de capas de filtros y funciones. se reduce la dimensionalidad de los datos y por ende son mas utiles para ser utilizados en una red neuronal tradicional. Para las CNN's se introducen 2 nuevos tipos de Neuronas artificiales enfocadas en reducir la dimensionalidad de las imagenes

##### Neurona Convolucional

En la fase de extracion de features (ie. Una *layer* de la CNN), las neuronas son remplazadas por operadores matriciales que realizan una operacion sobre la imagen. estos operadores contienen operadores convolucionales que tienen el objetivo de *resaltar* las caracterizticas de mayor importancia de la imagen. y asi reducir su dimensionalidad comprimiendo la informacion relevante

##### Neurona de Reduccion de Muestreo

Son neuronas encargadas de reducir el tamaño de la imagen tal que se resuman las caracteristicas de la imagen en una dimensionalidad menos a travez de metodos de pooling (metodos de reduccion de tamaño de muestras), en la siguiente imagen esto se puede ver mas claramente.


![Maxpool and Sampling](https://computersciencewiki.org/images/9/9e/MaxpoolSample.png)


## Areas a considerar

#### Reduccion de dimensionalidad
En el area del **Machine Learning** el tiempo de procesado es escencial, por ejemplo puede significar si un vehiculo autonomo reacciona correctamente frente a un obstaculo, ó en cosas mas banales como cuanto demora en cargar mi inicio en *netflix*.
Existe una relacion directa entre cantidad de *features* y el tiempo de procesado de algun modelo, es por esto que una gran área de la ciencia de datos y de ciencias de la computacion esta enfocada en reducir dimensionalidad de datos sin perder la calidad y representatividad de estos.

Existen diversos metodos de reduccion de dimensionalidad (*tambien llamado feature extraction*), y es muy importante investigar para obtener el mejor metodo para el modelo a desarollar. Por ahora presentaremos el metodo **PCA** (Analisis de Componentes Principales)

##### PCA & KPCA

El analisis de componentes principales se basa en describir los datos bajo conjuntos representativos. (Para ver su funcionamiento mas a detalle y las formulas asociadas ir a [PCA Wikipedia](https://es.wikipedia.org/wiki/An%C3%A1lisis_de_componentes_principales)).

Un ejemplo grafico de como funciona este algoritmo se puede observar al ejecutar los archivos *pca.py* y *kpca.py* en la carpeta Programs. basicamente en el programa pca se hace un analisis a datos ordenados obteniendo la varianza y desde ahi calcular el numero de dimensiones (*features*) utiles para el analisis. El caso de kpca o llamado Kernel-pca, primero se ha de realizar un analisis previo a los datos. este analisis se puede estudiar a detalle en [KPCA Wikipedia](https://en.wikipedia.org/wiki/Kernel_principal_component_analysis). y su funcionamiento en codigo y grafico en el archivo *kpca.py*
(Asegurarse de tener las librerias *numpy*, *sklearn* y *matplotlib* instaladas antes de ejecutar los archivos)

##### Metodos a considerar

* [ICA (Independent Component Analysis)](https://en.wikipedia.org/wiki/Independent_component_analysis)

* [LDA (Linear Discriminant Analysis)](https://en.wikipedia.org/wiki/Linear_discriminant_analysis)

* [LLE (Locally Linear Embedding )](https://cs.nyu.edu/~roweis/lle/papers/lleintro.pdf)

* [t-SNE (t-distributed Stochastic Neighbor Embedding)](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)


#### Regularizacion de los datos

Es importante la correcta seleccion de los datos mas representativos de nuestro objetivo para una correcta modelacion. (Asi se obtiene una estimacion cercana a los datos y no tan costosa) a continuacion la imagen muestra la diferencia grafica de ajustar una aproximacion con menos datos de los nescesarios, los nescesarios y si hay datos innecesarios (a este proceso se le conoce como model-fitting)

![Overfitting Example](https://media.geeksforgeeks.org/wp-content/uploads/20190523171704/overfitting_21.png)

Para solucionar los potenciales errores se realizan ajustes utilizando metodos de regresion, Los metodos mas utiles suelen ser la [Lasso Regression](https://es.wikipedia.org/wiki/LASSO_(estad%C3%ADstica)) ó [Ridge Regression](https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Ridge_Regression.pdf). Estos no tan solo son utiles para Machine o Deep Learning, sino que constituyen una parte esencial del modelamiento de problemas diaros y analisis estadistico. Por lo que es muy recomendable estudiarlos y aplicarlos. Para ejemplos y implementacion revisar el python notebook [LinearRegression-Lasso-Ridge-Regression](https://github.com/CarloGauss33/DL-Workshop/blob/master/Programs/LinearRegression-Lasso-Ridge-Regression.ipynb)



# Aplicacion en un clasificador de imagenes

Para este Workshop implementaremos un classificador de imagenes con alta fiabilidad basado en el paper hecho por Gao Huang, Zhuang Liu, Kilian Q. Weinberger y Laurens van der Maaten, cuyo link se encuentra aca [Paper](https://arxiv.org/abs/1608.06993).

¿Por que en este modelo y no en otros?, basicamente porque tiene una arquitecura robusta que da resultados cercanos al 95% de exactitud de prediccion, lo que es excelente para una aplicacion común.

Aunque primero, hemos de conocer como crear esta arquitectura de redes neuronales densas. Al realizar un modelo sobre imagenes, deberia ser claro que uno de nuestros primeros pensamientos ha de ser utilizar redes convolucionales, y eso es lo que haremos. En general una red covolucional tiene una estructura de layers dada como

**Input ->Convolution ->ReLU ->Convolution ->ReLU ->Pooling -> ReLU ->Convolution ->ReLU ->Pooling ->Fully Connected**

Para asi obtener una buena recoleccion de features y con ella.

Para realizar esto hemos de conocer las librerias que utilizaremos y luego aplicarlo creando una red neuronal arbitraria.

## Librerias

### Numpy
Una de las librerias mas usadas en python, permite la utilizacion de funciones matematicas con una gran optimizacion al ser una libreria basada en **C**, uno de los usos principales para Data Scientist es su gran optimizacion con arrays y operaciones matriciales. [Documentacion](https://docs.scipy.org/doc/numpy/reference/)

### SciPy
Libreria muy util (y Amplia) para analisis de datos. [Documentacion](http://scipy.github.io/devdocs/index.html)

### OpenCV
La mas basta libreria para computer vision, posee una gran variedad de filtros y de funciones de analisis para procesar imagenes, ademas posee clasificadores built-in para imagenes. [Documentacion](https://opencv.org/)

### Scikit-Learn
La mas amplia libreria sobre machine learning y optimizacion tanto en *R* como en *Python*. posee una amplia variedad de modelos *plug and play* principalmente enfocados en clasificadores y optimizadores del area de machine learning [Scikit-learn](https://scikit-learn.org/stable/documentation.html)

### Pandas
Libreria amplia para la gestion de bases de datos sobre python, muy facil de conectar con SQLite, servicios de Azure. y procesadores de texto built-in para casi todos los formatos de guardado de datos. [Documentation](https://pandas.pydata.org/pandas-docs/stable/)

### Matplotlib
Libreria para visualizacion de datos muy similar a *R* y a *Matlab* para python. [Documentacion](https://matplotlib.org/3.1.1/contents.html) 

### Tensorflow
Tensorflow es una libreria hecha por google, la cual se enfoca en la implementacion de algotitmos de aprendizaje automatica y altamente optimizada para operaciones de tensores y la aplicacion en operaciones sobre el TPU. [Documentacion](https://www.tensorflow.org/api_docs)

### Pytorch
Pytorch es una libreria de Deep Learning lanzada recientemente que soluciona las cosas molestas de las librerias como Tensorflow y Keras. ademas es facil de utilizar y de conectar para procesos sobre la GPU y TPU. [Documentacion](https://pytorch.org/docs/stable/index.html)

## Creando Tu Primera Red Neuronal en python :eyes:

Utilizaremos Pytorch para crear una red neuronal sencilla con un par de layers convolucionales y un par de layers lineales.
Tendra la siguiente estructura:

**Input -> Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> View -> Linear -> ReLU -> Linear -> ReLU -> Linear -> MSELoss -> Loss**

El contenido para crearlo esta explicado y realizado en el archivo *SimpleCNN.ipynb* en la carpeta Programs, [Link]()