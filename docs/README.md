PONER INDEX

## Que es la Inteligencia Artificial

Para poder explicar cualquier cosa sobre deep learning y por que es tan importante, se ha de partir sobre la base de que es la Inteligencia artificial. la Inteligencia Artificial (tambien llamada inteligencia de computadoras ó IA) es inteligencia expresada por maquinas, mas formalmente se define como:
*"La habilidad de un sistema (computador) para interpretar correctamente datos externos, aprender de estos, y usar el aprendizaje para conseguir objetivos y tareas especificas a travez de una adaptacion flexible"*
En otras palabras la **IA** se refiere a algoritmos que permiten a un computador alimentarse de datos y obtener patrones sobre estos, sin estos ser programados en la maquina.

Una IA tipica analiza su entorno y realiza cambios para obtener un mejor resultado. para esto posee una *Utility Function* o tambien llamada variable objetivo, la cual puede ser explicitamente programada en la maquina o tambien inducida a travez de los datos. Luego a travez de la utilizacion de diversos algoritmos, se analiza el conjunto de datos. para que al final si se ingresa un dato similar (No igual) a estos, el programa con el aprendizaje previo predice que este dato pertenece a un conjunto que ya conoce.

De forma mas directa y a modo de ejemplo, un vehiculo autonomo es conducido por una persona durante mucho tiempo en muchas carreteras distintas, luego del proceso de aprendizaje si el auto (sin piloto), se enfrenta a una carretera de caracterizticas similares a las anteriores (una curva por ejemplo), va a poder ejecutar la accion requerida (virar si es una curva).

## DEEP LEARNING 

### Que es el Deep Learning

Deep learning o *Aprendizaje profundo* son conjuntos de algoritmos basados en **redes neuronales artificiales**, en palabras comunes es una tecnica del area del aprendizaje de maquina, que trata de simular el aprendizaje humano, esto es aprender con el ejemplo. 
La tecnica de Deep Learning es la responsable de los vehiculos autonomos, del reconocimiento facial y el control por voz en dispositivos tecnologicos. Haciendo que el publico general se interese cada vez mas en este fascinante mundo de la Inteligencia Artificial.

Para implementar la tecnica de Deep Learning primero hemos de conocer sus elementos, partiendo por su elemento mas basico. la Neurona

#### Neurona
La neurona artificial (Conocida como *Neurona de McCulloch-Pitts*) es una unidad de calculo que trata de imitar el comportamiento de una neurona natural. basicamente la neurona realiza una operacion de suma ponderada de sus entradas. y luego se aplica una funcion no lineal dando asi un resultado llamado output. (para motivos pedagogicos se puede interpretar una neurona como una funcion de n entradas que las multiplica por *pesos* y luego a su suma aplica una funcion no lineal llamada *funcion de activacion*)

El diseño de una neurona se puede ver
![Neurona de McCulloch-Pitts](https://upload.wikimedia.org/wikipedia/commons/c/c8/Mccullochpitts.png)

y la operacion matematica se veria como

*f*(*x*<sub>1</sub>, *x*<sub>2</sub>, …, *x*<sub>*n*</sub>) = *g*(*c*<sub>1</sub>*x*<sub>1</sub> + *c*<sub>2</sub>*x*<sub>2</sub> + … + *c*<sub>*n*</sub>*x*<sub>*n*</sub> − *d*)

Con:
* f la *funcion* que representa el output de la neurona
* \[x_i\] las entradas de la neurona
* \[c_i\] los pesos de las entradas
* g la funcion de activacion
* d el umbral de la operacion

Estas operaciones se pueden representar facilmente con matrizes y asi facilitar la implementacion de estas en programas de computadoras

#### Red Neuronal
Una red Neuronal son conjuntos de capas (*layers*) de neuronas artificiales conectadas entre si para transmitir señales. la señal de input se transmite sucesivamente aplicando la operacion correspondiente y luego de un numero de ciclos se genera una salida. 

