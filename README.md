# Implementacion ModelosML validación cruzada en AWS


Este repositorio contiene la Implementación, publicación, uso y eleminacón de Modelos de Machine learning mediante validación cruzada en Infraestructura AWS

---------.
# Contenido

1.  [Preparación de datos para validación cruzada usando Apache Spark en Amazon Athena](#preparación-de-datos-para-validación-cruzada)
    
2.  [Validación cruzada e hiperparametrización de un modelo de Regresión Lineal en Amazon SageMaker](#validación-cruzada-e-hiperparametrización-de-regresión-lineal)
3.  [Registro de Modelos](#registro-del-modelos)
4.  [Publicación Deployment](#publicación-*deployment)
5.  [Uso del Modelo](#uso-del-modelo)
6.  [Uso del Modelo de Forma Remota](#uso-del-modelo-de-forma-remota)
    

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

# 2. Validación cruzada e hiperparametrización de un modelo de Regresión Lineal en Amazon SageMaker

Inicialmente se deben entrenar cada uno de los conjuntos de datasets generados, para nuestro caso de sebe realizar el entrenamiento por separado para cada uno de los 5 conjuntos de data frames. Con la anterior se determian cual modelo tiene el menor MSE.

```python
#Utilitario para leer archivos de datos
from sagemaker.inputs import TrainingInput

#Bucket en donde se encuentran los archivos
#IMPORTANTE: REEMPLAZAR "XXX" POR TUS INICIALES
bucket = "datasetsbdajac"

#Lectura de datos de entrenamiento
dataTrain = TrainingInput(
    f"s3://{bucket}/data/insurance_dataset_validacion_cruzada/vc1/train/", #Ruta del archivo
    content_type = "text/csv", #Formato del archivo
    distribution = "FullyReplicated", #El archivo será copiado en todos los servidores
    s3_data_type = "S3Prefix", #Desde donde se lee el archivo (S3)
    input_mode = "File", #Los registros se encuentran dentro de archivos
    record_wrapping = "None" #Envoltorio de optimización
)

#Lectura de datos de validación
dataTest = TrainingInput(
    f"s3://{bucket}/data/insurance_dataset_validacion_cruzada/vc1/test/", #Ruta del archivo
    content_type = "text/csv", #Formato del archivo
    distribution = "FullyReplicated", #El archivo será copiado en todos los servidores
    s3_data_type = "S3Prefix", #Desde donde se lee el archivo (S3)
    input_mode = "File", #Los registros se encuentran dentro de archivos
    record_wrapping = "None" #Envoltorio de optimización
)

#Importamos el utilitario para definir el entrenador del algoritmo
from sagemaker.estimator import Estimator

#Definimos el entrenador del algoritmo
entrenador = Estimator(
    image_uri = sagemaker.image_uris.retrieve("linear-learner", region), #Descargamos la implementación del algoritmo desde la región donde trabajamos
    role = rol, #Rol que ejecuta servicios sobre AWS
    instance_count = 1, #Cantidad de servidores de entrenamiento
    instance_type = "ml.m5.large", #Tipo de servidor de entrenamiento
    predictor_type = "regressor", #Tipo de predicción del algoritmo
    sagemaker_session = sesion, #Sesión de SageMaker
    base_job_name = "entrenamiento-prediccion-numerica-vc1" #Nombre del job de entrenamiento
)

#Configuramos los parametros del algoritmo
entrenador.set_hyperparameters(
    feature_dim = 11, #Cantidad de features
    predictor_type = "regressor", #Indicamos que tipo de predicción es
    normalize_data = "true", #Normalizamos los features
    normalize_label = "true" #Normalizamos el label
)

#Entrenamos y validamos el modelo
#MIENTRAS SE ENTRENA EL MODELO: En SageMaker, en la sección "Jobs", en la opción "Training" podemos ver cómo el modelo se entrena
#TIEMPO DE ENTRENAMIENTO: 5 MINUTOS
entrenador.fit({"train": dataTrain, "validation": dataTest})
```

## Configuración del Entorno

Se utiliza Amazon SageMaker para el entrenamiento y la hiperparametrización.
Se importan las librerías necesarias de SageMaker.

```python
import sagemaker

#Iniciamos sesión en el servicio de SageMaker
sesion = sagemaker.Session()

#Obtenemos la ejecución en donde estamos trabajando
region = sesion.boto_region_name

#Verificamos
print(region)

#Obtenemos el rol de ejecución de SageMaker
rol = sagemaker.get_execution_role()
```

## Lectura de Datos

Se utilizan TrainingInput para leer los datos de entrenamiento y validación desde S3.

```python
from sagemaker.inputs import TrainingInput

bucket = "datasetsbdatest001"

dataTrain = TrainingInput(
    f"s3://{bucket}/data/insurance_dataset_validacion_cruzada/vc1/train/",
    content_type = "text/csv",
    distribution = "FullyReplicated",
    s3_data_type = "S3Prefix",
    input_mode = "File",
    record_wrapping = "None"
)

dataTest = TrainingInput(
    f"s3://{bucket}/data/insurance_dataset_validacion_cruzada/vc1/test/",
    content_type = "text/csv",
    distribution = "FullyReplicated",
    s3_data_type = "S3Prefix",
    input_mode = "File",
    record_wrapping = "None"
)
```

## Configuración del Algoritmo

Se utiliza el algoritmo "linear-learner" de SageMaker.
Se configura un Estimator con los parámetros básicos del modelo.

```python
from sagemaker.estimator import Estimator

entrenador = Estimator(
    image_uri = sagemaker.image_uris.retrieve("linear-learner", region),
    role = rol,
    instance_count = 1,
    instance_type = "ml.m5.large",
    predictor_type = "regressor",
    sagemaker_session = sesion,
    base_job_name = "entrenamiento-prediccion-numerica-vc1"
)

entrenador.set_hyperparameters(
    feature_dim = 11,
    predictor_type = "regressor",
    normalize_data = "true",
    normalize_label = "true"
)
```

## Definición de Hiperparámetros

Se definen rangos para los hiperparámetros:

learning_rate: [0.0001, 0.001, 0.01, 0.1]
l1: [0.001, 0.01, 0.1]


Se utiliza HyperparameterTuner para configurar la búsqueda de hiperparámetros.

```python
from sagemaker.tuner import CategoricalParameter

hyperparametros = {
    "learning_rate": CategoricalParameter([0.0001, 0.001, 0.01, 0.1]),
    "l1": CategoricalParameter([0.001, 0.01, 0.1])
}
```

## Entrenamiento del Modelo

Se ejecuta la malla de hiperparámetros con mallaDeHyperParametros.fit().
Se entrenan múltiples modelos con diferentes combinaciones de hiperparámetros.

## Selección del Mejor Modelo

Se obtiene el mejor modelo basado en la métrica de validación (MSE).
Se extraen las métricas y los hiperparámetros del mejor modelo.

```python
# Agregar índice a los datos
dfIndice = dfDataset.withColumn("indice_fila", f.monotonically_increasing_id())

# Crear índice ordenado
dfIndice = dfIndice.withColumn(
    "indice_fila_2", 
    f.row_number().over(Window.orderBy("indice_fila"))
)

# Dividir datos en 5 partes
cantidadDeRegistrosValidacion = int(numeroDeRegistros/5)

# Crear DataFrames para cada fold
df1 = dfIndice.filter(
    (dfIndice["indice_fila_2"] >= 0) &
    (dfIndice["indice_fila_2"] < cantidadDeRegistrosValidacion)
).drop("indice_fila").drop("indice_fila_2")

# ... (repetir para df2, df3, df4, df5)
```
## Hiperparametrizción

```pyhton
# Definir hiperparámetros
hyperparametros = {
    "learning_rate": CategoricalParameter([0.0001, 0.001, 0.01, 0.1]),
    "l1": CategoricalParameter([0.001, 0.01, 0.1])
}

# Configurar malla de hiperparámetros
mallaDeHyperParametros = HyperparameterTuner(
    entrenador,
    "validation:mse",
    hyperparametros,
    objective_type = "Minimize",
    max_jobs = 12,
    max_parallel_jobs = 10
)

# Entrenar modelos
mallaDeHyperParametros.fit(inputs = {"train": dataTrain, "validation": dataTest})

# Obtener mejor modelo
nombreDelMejorModelo = mallaDeHyperParametros.best_training_job()

# Extraer métricas y hiperparámetros
descripcionDeEntrenamiento = sagemakerCliente.describe_training_job(TrainingJobName = nombreDelMejorModelo)
metricas = descripcionDeEntrenamiento["FinalMetricDataList"]
hiperparametros = descripcionDeEntrenamiento["HyperParameters"]
```

## Selección del mejor modelo:

```python
import boto3

sagemakerCliente = boto3.client("sagemaker")

nombreDelMejorModelo = mallaDeHyperParametros.best_training_job()

print(nombreDelMejorModelo)
```

## Extracción de estadísticas y parámetros del modelo:

```python
descripcionDeEntrenamiento = sagemakerCliente.describe_training_job(TrainingJobName = nombreDelMejorModelo)

print(descripcionDeEntrenamiento["FinalMetricDataList"])

for metrica in descripcionDeEntrenamiento["FinalMetricDataList"]:
    if metrica["MetricName"] == "validation:mse":
        print(metrica["Value"])

print(descripcionDeEntrenamiento["HyperParameters"])

print(descripcionDeEntrenamiento["HyperParameters"]["learning_rate"])
print(descripcionDeEntrenamiento["HyperParameters"]["l1"])
```

# 3. Registro de Modelos

Este proceso permite guardar y versionar el modelo entrenado, facilitando su gestión y despliegue en el futuro. El registro del modelo incluye metadatos importantes como los tipos de datos aceptados y los requisitos de infraestructura, lo que facilita su uso y mantenimiento en un entorno de producción.

```python
#Registramos el modelo
registroDelModelo = modelo.register(
    model_package_group_name = nombreDelModelo,
    content_types = tiposDeRegistrosInput, #Tipo de registros INPUT del modelo
    response_types = tiposDeRegistrosOutput, #Tipo de registros OUTPUT del modelo
    inference_instances = tipoDeInstanciasDeEjecucion, #Tipo de servidor en donde se colocará el modelo
    transform_instances = tipoDeInstanciasDeEjecucion #Tipo de servidor en donde el modelo realizará cálculos intermedios
)
```
# 4. Publicacion Deployment

Este proceso permite desplegar el modelo entrenado como un servicio web accesible a través de un endpoint. Esto facilita la integración del modelo en aplicaciones y sistemas que necesiten realizar predicciones en tiempo real.
Puntos importantes a destacar:

El despliegue del modelo puede tomar varios minutos (aproximadamente 5 minutos según el comentario en el código).
Se utiliza una instancia ml.m5.large para el despliegue, lo cual es adecuado para cargas de trabajo moderadas.
El modelo se despliega inicialmente con una sola instancia, pero esto puede escalarse según las necesidades.
El endpoint creado proporciona un punto de acceso para realizar predicciones utilizando el modelo desplegado.



```python
#Definimos el nombre del entrenamiento al que nos conectamos
nombreDeEntrenamiento = "XXXXXXXXXXXXXXXXXXXXXXX"

#Definimos el algoritmo que usamos para entrenar el modelo
algoritmo = "linear-learner"

#Nos conectamos al servicio de SageMaker
sagemakerCliente = boto3.client("sagemaker")

#Obtenemos la descripción del entrenamiento
descripcionDeEntrenamiento = sagemakerCliente.describe_training_job(TrainingJobName = nombreDeEntrenamiento)

#Obtenemos la ruta en donde el modelo se encuentra almacenado
rutaDelModelo = descripcionDeEntrenamiento["ModelArtifacts"]["S3ModelArtifacts"]

#Verificamos
print(rutaDelModelo)

#Utilitario para leer modelos
from sagemaker.model import Model
#Leemos el modelo
modelo = Model(
    model_data = rutaDelModelo, #Ruta del modelo
    role = rol, #Rol de ejecución
    image_uri = sagemaker.image_uris.retrieve(algoritmo, region), #Descargamos la implementación del algoritmo desde la región donde entrenamos
    sagemaker_session = sesion #Sesión de SageMaker
)
```

### Despliegue

```python
#Desplegamos el modelo
#TIEMPO: 5 MINUTOS
modelo.deploy(
    initial_instance_count = cantidadInicialDeInstancias, #Cantidad de servidores
    instance_type = tipoDeInstanciaDeEndpoint, #Tipo de servidor
    endpoint_name = nombreDelEndpoint #Nombre del punto de acceso al modelo
)
```

# 5. Uso del Modelo

Este enfoque permite utilizar de manera eficiente un modelo de machine learning desplegado en SageMaker, facilitando la integración de predicciones en aplicaciones y flujos de trabajo de análisis de datos.

## Puntos importantes a destacar:

El código maneja la lectura de múltiples archivos CSV desde S3, lo que es útil para conjuntos de datos grandes.
Se utiliza la biblioteca pandas para el manejo eficiente de los datos.
La serialización y deserialización de datos se manejan automáticamente con las clases CSVSerializer y JSONDeserializer.
El código está preparado para manejar grandes volúmenes de datos, realizando predicciones en lote.

```python
#Endpoint de acceso al modelo
nombreDelEndpoint = "endpoint-numerico-XXX"

#Utilitario para usar el modelo
from sagemaker.predictor import Predictor

#Utilitario para serializar el INPUT del modelo (CSV)
from sagemaker.serializers import CSVSerializer

#Utilitario para serializar el OUTPUT del modelo (JSON)
from sagemaker.deserializers import JSONDeserializer

#Creamos un predictor para el modelo desplegado
predictor = Predictor(
    endpoint_name = nombreDelEndpoint, #Nombre del endpoint
    sagemaker_session = sesion, #Sesión de SageMaker
    serializer = CSVSerializer(), #Serializador que envía los datos al modelo
    deserializer = JSONDeserializer() #Des-serializador que extrae la respuesta del modelo
)

#Serializamos los registros
registrosSerializados = CSVSerializer().serialize(matrizDeRegistros)

#Usamos el modelo para hacer predicciones
resultados = predictor.predict(registrosSerializados)

#Verificamos los resultados
#Notemos que como no hemos calibrado el modelo, el modelo está entregando malos resultados
resultados
```

Este enfoque de despliegue como microservicio permite una fácil gestión, escalabilidad y mantenimiento del modelo en un entorno de producción.

# 6.  Uso del Modelo de forma Remota
Este enfoque es particularmente útil en escenarios donde se necesita integrar las predicciones del modelo en aplicaciones o servicios externos, permitiendo un acceso rápido y eficiente al modelo desplegado en SageMaker sin la necesidad de implementar toda la infraestructura de machine learning en el entorno de la aplicación.

```python
# Datos de entrada para la predicción (ajusta según tus requisitos)
registros = [
    [
        19.0, #age
        27.9, #bmi
        0.0, #children
        1, #sex_female
        0, #sex_male
        0, #region_northeast
        0, #region_northwest
        0, #region_southeast
        1, #region_southwest
        0, #smoker_no
        1 #smoker_yes
    ],
    [
        18.0, #age
        33.770, #bmi
        1.0, #children
        0, #sex_female
        1, #sex_male
        0, #region_northeast
        0, #region_northwest
        1, #region_southeast
        0, #region_southwest
        1, #smoker_no
        0 #smoker_yes
    ],
    [
        28.0, #age
        33.000, #bmi
        3.0, #children
        0, #sex_female
        1, #sex_male
        0, #region_northeast
        0, #region_northwest
        1, #region_southeast
        0, #region_southwest
        1, #smoker_no
        0 #smoker_yes
    ]
]

#Serializamos los registros
registrosSerializados = CSVSerializer().serialize(registros)

"""# 5. Uso del Modelo"""

#Endpoint de acceso al modelo
#IMPORTANTE: En "XXX" colocar la fecha de hoy, hay un bug que hace que no puedas ver tu modelo si previamente ya lo creaste y borraste
nombreDelEndpoint = "endpoint-numerico-XXX"

#Nos conectamos al cliente de ejecución remota de modelos
sagemakerRuntime = boto3.client("sagemaker-runtime")

#Nos conectamos al end-point para obtener la predicción
respuesta = sagemakerRuntime.invoke_endpoint(
    EndpointName = nombreDelEndpoint,
    ContentType = "text/csv",
    Body = registrosSerializados
)

#Extraemos la respuesta de la petición
resultados = respuesta["Body"].read()

#Verificamos los resultados de las predicciones
resultados
```
Puntos importantes a destacar:

Este enfoque permite utilizar el modelo de forma remota, sin necesidad de cargar el modelo completo en el entorno local.
La serialización de datos se maneja manualmente con CSVSerializer, lo que permite un control preciso sobre el formato de los datos enviados al modelo.
El uso de boto3 para interactuar con el endpoint de SageMaker permite una integración más directa con los servicios de AWS.
Este método es eficiente para realizar predicciones en tiempo real o para procesar pequeños lotes de datos.




