# Implementacion ModelosML validación cruzada en AWS
Implementación, publicación, uso y eleminacón de Modelos de Machine learning mediante validación cruzada en Infraestructura AWS
Contenido
---------

Este repositorio contiene la implementación de modelos de machine learning utilizando validación cruzada en AWS. Se enfoca en 5 aspectos principales:

1.  [Preparación de datos para validación cruzada usando Apache Spark en Amazon Athena](#preparación-de-datos-para-validación-cruzada)
    
2.  [Validación cruzada e hiperparametrización de un modelo de Regresión Lineal en Amazon SageMaker](#validación-cruzada-e-hiperparametrización-de-regresión-lineal)
    

#  1. Preparación de Datos para Validación Cruzada
## Entorno y Configuración

Se utiliza un Notebook PySpark en Amazon Athena.
Se importan las librerías necesarias, incluyendo PySpark SQL y Pandas.

```python
#Creamos un Notebook en PySpark sobre Athena

# Utilitarios para modificar el esquema de metadatos
from pyspark.sql.types import StructType, StructField

#Importamos los tipos de datos que definiremos para cada campo
from pyspark.sql.types import StringType, IntegerType, DoubleType

#Importamos la librerIa de pandas compatible con entornos de clUster de Big Data
import pyspark.pandas as pd

#Por defecto un dataframe Pandas muestra 1000 registros
#Vamos a indicarle que solo muestre 20 para que no se sature el notebook
pd.set_option("display.max_rows", 20)

#Libreria para manipular los servicios de AWS
import boto3

#Libreria utilitaria para JSON
import json
```

## Lectura de Datos

Los datos se leen de un archivo CSV almacenado en Amazon S3.
Se utiliza la función spark.read.format("csv") para leer el archivo.

```python
#Leemos los datos
#No es necesario indicar el esquema de metadatos, ya que no vamos a procesar los campos
#Solo vamos a crear los cortes del dataframe para la validación cruzada
#IMPORTANTE: Notar que los archivos ya no tienen header
#IMPORTANTE: Reemplazar "XXX" por tus iniciales
dfDataset = spark.read.format("csv").option("header", "false").option("delimiter", ",").option("encoding", "ISO-8859-1").load("s3://datasetsbdatestXXX/data/insurance_dataset/")

#Vemos el esquema de metadatos
dfDataset.printSchema()

#Verificamos
dfDataset.show()
```

## Procesamiento de Datos

Se agrega una columna de índice a los datos usando monotonically_increasing_id().
Se utiliza Window.orderBy() para crear un índice ordenado y consistente.
Se calcula el número total de registros y se divide en 5 partes iguales para la validación cruzada.

```python
#Utilitarios de Spark
import pyspark.sql.functions as f

#Agregamos una columna que indique el �ndice de la fila
dfIndice = dfDataset.withColumn("indice_fila", f.monotonically_increasing_id())

#Para solucionar lo anterior usaremos el utilitario "Window"
#Este utilitario permite definir una columna para ordenar los registros
#Los ordenaremos por la columna "indice_fila"
from pyspark.sql.window import Window

#Agregamos la columna "indice_fila_2"
#Usamos nuevamente la funci�n "row_number" para agregar el �ndice
#Pero esta vez se generar� en orden, ya que estamos usando el Window.orderBy
dfIndice = dfIndice.withColumn(
    "indice_fila_2", 
    f.row_number().over(Window.orderBy("indice_fila"))
)

#Obtenemos el n�mero total de registros
numeroDeRegistros = dfIndice.count()

#Calculamos cu�ntos registros representan el 20%
cantidadDeRegistrosValidacion = int(numeroDeRegistros/5)
```

## Creación de Conjuntos de Datos

Se crean 5 DataFrames (df1, df2, df3, df4, df5) representando cada fold de la validación cruzada.
Cada DataFrame contiene el 20% de los datos originales.

```python
#Obtenemos todos los cortes

#PRIMER CORTE
df1 = dfIndice.filter(
    (dfIndice["indice_fila_2"] >= 0) &
    (dfIndice["indice_fila_2"] < cantidadDeRegistrosValidacion)
).drop("indice_fila").drop("indice_fila_2")

#SEGUNDO CORTE
df2 = dfIndice.filter(
    (dfIndice["indice_fila_2"] >= cantidadDeRegistrosValidacion) &
    (dfIndice["indice_fila_2"] < 2*cantidadDeRegistrosValidacion)
).drop("indice_fila").drop("indice_fila_2")

#TERCER CORTE
df3 = dfIndice.filter(
    (dfIndice["indice_fila_2"] >= 2*cantidadDeRegistrosValidacion) &
    (dfIndice["indice_fila_2"] < 3*cantidadDeRegistrosValidacion)
).drop("indice_fila").drop("indice_fila_2")

#CUARTO CORTE
df4 = dfIndice.filter(
    (dfIndice["indice_fila_2"] >= 3*cantidadDeRegistrosValidacion) &
    (dfIndice["indice_fila_2"] < 4*cantidadDeRegistrosValidacion)
).drop("indice_fila").drop("indice_fila_2")

#QUINTO CORTE
df5 = dfIndice.filter(
    (dfIndice["indice_fila_2"] >= 4*cantidadDeRegistrosValidacion) &
    (dfIndice["indice_fila_2"] <= 5*cantidadDeRegistrosValidacion)
).drop("indice_fila").drop("indice_fila_2")
```

## Almacenamiento de Datos

Los datos se almacenan en S3 en formato CSV.
Se crean directorios separados para datos de entrenamiento y prueba para cada fold.
Se utiliza la función write.format("csv") para guardar los DataFrames.

```python
#Nombre del bucket desde donde se lee el archivo
#IMPORTANTE: REEMPLAZAR "XXX" POR TUS INICIALES
bucket = "datasetsbdatestXXX"

#Directorio del dataset
directorioDataset = "data/insurance_dataset_validacion_cruzada/vc1"

#Directorio en donde est�n los archivos de entrenamiento
directorioDeEntrenamiento = f"s3://{bucket}/{directorioDataset}/train/"

#Directorio en donde est�n los archivos de validaci�n
directorioDeValidacion = f"s3://{bucket}/{directorioDataset}/test/"

#Almacenamos el dataframe de entrenamiento
dfTrain.write.format("csv").option("header", "false").option("delimiter", ",").option("encoding", "ISO-8859-1").mode("overwrite").save(directorioDeEntrenamiento)

#Almacenamos el dataframe de validación
dfTest.write.format("csv").option("header", "false").option("delimiter", ",").option("encoding", "ISO-8859-1").mode("overwrite").save(directorioDeValidacion)
```

## Limpieza

Se eliminan archivos temporales "_SUCCESS" utilizando boto3.

```python
#Nos conectamos al servicio de "S3" para eliminar los archivos "_SUCCESS"
s3 = boto3.client("s3")

#Eliminamos el archivo "_SUCCESS" del dataset de entrenamiento
s3.delete_object(
    Bucket = bucket,
    Key = f"{directorioDataset}/train/_SUCCESS"
)

#Eliminamos el archivo "_SUCCESS" del dataset de validaci�n
s3.delete_object(
    Bucket = bucket,
    Key = f"{directorioDataset}/test/_SUCCESS"
)
```

## Función Utilitaria



Se crea una función generar_dataset_validacion_cruzada() para automatizar el proceso de generación de datasets para cada fold de la validación cruzada.

```python
#Función utilitaria
def generar_dataset_validacion_cruzada(dfTest, dfTrain, bucket, directorioDataset):
    #Directorio en donde están los archivos de entrenamiento
    directorioDeEntrenamiento = f"s3://{bucket}/{directorioDataset}/train/"

    #Directorio en donde están los archivos de validación
    directorioDeValidacion = f"s3://{bucket}/{directorioDataset}/test/"
```
    
#Almacenamos el dataframe de entrenamiento
Se utiliza la función utilitaria para generar 5 conjuntos de datos de validación cruzada, cada uno con sus respectivos datos de entrenamiento y prueba.

```python
#Generamos todos los datasets

#Generamos el PRIMER conjunto de datasets
generar_dataset_validacion_cruzada(
    df1,
    df2.union(df3).union(df4).union(df5),
    bucket,
    "data/insurance_dataset_validacion_cruzada/vc1"
)

#Generamos el SEGUNDO conjunto de datasets
generar_dataset_validacion_cruzada(
    df2,
    df1.union(df3).union(df4).union(df5),
    bucket,
    "data/insurance_dataset_validacion_cruzada/vc2"
)

#Generamos el TERCER conjunto de datasets
generar_dataset_validacion_cruzada(
    df3,
    df1.union(df2).union(df4).union(df5),
    bucket,
    "data/insurance_dataset_validacion_cruzada/vc3"
)

#Generamos el CUARTO conjunto de datasets
generar_dataset_validacion_cruzada(
    df4,
    df1.union(df2).union(df3).union(df5),
    bucket,
    "data/insurance_dataset_validacion_cruzada/vc4"
)

#Generamos el QUINTO conjunto de datasets
generar_dataset_validacion_cruzada(
    df5,
    df1.union(df2).union(df3).union(df4),
    bucket,
    "data/insurance_dataset_validacion_cruzada/vc5"
)
```


   



