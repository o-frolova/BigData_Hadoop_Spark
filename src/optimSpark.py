# -*- coding: utf-8 -*-
import time
import numpy as np
from tqdm import tqdm
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressionModel
import matplotlib.pyplot as plt
from functions import *
from parsing_args import *

def preprocess_data(partition, df, model_rf):
        df = preprocessing_data(df)
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        df = assembler.transform(df)
        predictions = model_rf.transform(df)
        return predictions
        
if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("ParallelSparkApp") \
        .master(args.spark_url) \
        .getOrCreate()

    sc = spark.sparkContext
    logger = spark.sparkContext._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)

    total_time = []
    total_RAM = []
    for i in tqdm(range(100)):
        start_time = time.time()

        model_rf = RandomForestRegressionModel.load(args.model_path)
        df = spark.read.csv(args.data_path, header=True)
        predictions = spark.sparkContext.parallelize(range(100)).map(preprocess_data)

        end_time = time.time()
        total_time.append(end_time - start_time)
        total_RAM.append(get_executor_memory(sc))

    draw_graph(total_time, total_RAM, './optim.png')

    print()
    print('Average memory(MB):', np.mean(total_RAM))
    print('Аverage time(c)', np.mean(total_time))
    spark.stop()