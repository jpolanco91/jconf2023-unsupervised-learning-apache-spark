# JConf 2023: Unsupervised Learning con Java, Apache Spark y MLlib.

## Descripción del proyecto.

Este proyecto es el utilizado para ilustrar los conceptos de la charla Unsupervised Learning con Java, Apache Spark y MLlib en el evento JConf 2023 celebrado en Santo Domingo, Republica Dominicana el 23 de Julio 2023.

Este es el dataset opensource utilizado para el mismo: https://www.kaggle.com/datasets/sriramm2010/uci-bike-sharing-data


## Instalando la aplicación y sus dependencias.


### Clonando el codigo.

Para clonar el codigo instalamos git en nuestra maquina y luego ejecutamos este comando:

`git clone: git@github.com:jpolanco91/jconf2023-unsupervised-learning-apache-spark.git`

##### *Instalando el JDK*

Para poder correr, modificar y ejecutar el codigo necesitaremos tener el Java Development Kit (JDK) instalado. Para este proyecto utilizaremos el JDK version 20 (que es el que esta especificado en el archivo de dependencias de Maven, el pom.xml). Para descargarlo e instalarlo podemos ir a este enlace: https://www.oracle.com/java/technologies/javase/jdk20-archive-downloads.html

##### *Instalando Maven*

Como pre-requisito para poder instalar las dependencias y también compilar y correr el codigo debemos instalar Maven. Podemos descargar e instalar maven siguiendo estos pasos aquí: https://maven.apache.org/download.cgi y https://maven.apache.org/install.html

##### *Instalacion de dependencias*

1. Abrimos la terminal de nuestro sistema (Terminal en macOS y Linux, command prompt (cmd) o Windows Powershell en Windows) y luego nos cambiamos a la carpeta que se creo al clonar el repositorio: `cd /ruta/a/carpeta/proyecto`
2. Una vez dentro de la carpeta del proyecto ejecutamos el siguiente comando: `mvn clean install`. Esto instalara las dependencias del `pom.xml`, limpiara cualquier `.class precompilado y luego compilara el codigo de la app.
3. A pesar de que el codigo este ya compilado con el paso anterior, procederemos a re-compilarlo con este otro comando: `mvn clean compile assembly:single` para que así todas las dependencias que se han descargado sean incluidas en el JAR final que se forma al compilarlo, de manera que no les de un error por falta de dependencias. Cada vez que desarrollemos o tengamos que correr el codigo este es el comando que utilizaremos para compilar el codigo antes de correrlo.

#### *Corriendo el codigo*

 1. En nuestra terminal le otorgamos permiso de ejecucion al archivo `run.sh` el cual es el script que nos permitirá correr el codigo con todos los parámetros requeridos por Java mediante el siguiente comando: `chmod a+x run.sh`.
 2. Para correr el codigo utilizamos el siguiente comando: `./run.sh`
