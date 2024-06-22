from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.sql.functions import isnull,col
import psutil
import matplotlib.pyplot as plt

feature_columns = ['user_id', 'price', 'count', 'Sum_req', 'region_name_encoded',
                   'cpe_manufacturer_name_encoded', 'cpe_type_cd_encoded', 'PartofDay_encoded']

def full_nan_value_pyspark(dataset):
    return dataset.na.fill(0)

def convert_value_to_int_pyspark(dataset):
    # Convert each column to IntegerType
    columns = ['age', 'user_id', 'price', 'count','Sum_req']
    for column in columns:
        dataset = dataset.withColumn(column, dataset[column].cast(IntegerType()))
    return dataset

def removing_unnecessary_columns_pyspark(dataset):
    # List of columns to delete
    columns_to_drop = ['city_name', 'cpe_model_os_type', 'cpe_model_name', 'region_name', 'cpe_manufacturer_name', 'cpe_type_cd', 'PartofDay']
    
    dataset = dataset.drop(*columns_to_drop)
    return dataset

def concert_categorical_data_pyspark(dataset):
    categorical_features = ['region_name', 'cpe_manufacturer_name', 'cpe_type_cd', 'PartofDay']

    # Create a list of indexes
    indexers = [StringIndexer(inputCol=column, outputCol=column + "_index") for column in categorical_features]
    
    # Create a list of encoders
    encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(), outputCol=column + "_encoded") for indexer, column in zip(indexers, categorical_features)]
    
    # Combine all the transformations into one pipeline
    pipeline = Pipeline(stages=indexers + encoders)
    
    model = pipeline.fit(dataset)
    dataset = model.transform(dataset)
    
    # Remove original and indexed columns, leaving only coded columns
    for column in categorical_features:
        dataset = dataset.drop(column)
        dataset = dataset.drop(column + "_index")
    
    return dataset

def preprocessing_data(data):
  data = full_nan_value_pyspark(data)
  data = concert_categorical_data_pyspark(data)
  data = removing_unnecessary_columns_pyspark(data)
  data = convert_value_to_int_pyspark(data)
  data = data.filter(col("price").isNotNull())
  return data

def get_executor_memory(sc):
    executor_memory_status = sc._jsc.sc().getExecutorMemoryStatus()
    executor_memory_status_dict = sc._jvm.scala.collection.JavaConverters.mapAsJavaMapConverter(executor_memory_status).asJava()
    total_used_memory = 0
    for executor, values in executor_memory_status_dict.items():
        total_memory = values._1() / (1024 * 1024)  # Convert bytes to MB
        free_memory = values._2() / (1024 * 1024)    # Convert bytes to MB
        used_memory = total_memory - free_memory
        total_used_memory += used_memory
    return total_used_memory

def draw_graph(total_time, total_RAM, path_name):
  plt.figure(figsize=(14, 6))
  plt.subplot(1, 2, 1)
  plt.hist(total_time, bins=30, edgecolor='k', alpha=0.7)
  plt.xlabel('Time(c)')
  plt.ylabel('Frequency')
  plt.title('Histogram of time distribution')
  plt.grid(True)

  plt.subplot(1, 2, 2)
  plt.hist(total_RAM, bins=30, edgecolor='k', alpha=0.7)
  plt.xlabel('RAM(MB)')
  plt.ylabel('Frequency')
  plt.title('Histogram of RAM distribution')
  plt.grid(True)

  plt.tight_layout()
  plt.savefig(path_name)