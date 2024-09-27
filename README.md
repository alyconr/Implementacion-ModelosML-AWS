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
