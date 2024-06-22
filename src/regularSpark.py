from pyspark.sql import SparkSession
import psutil
import os
import time
from tqdm import tqdm
import numpy as np
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressionModel
from functions import *
from parsing_args import *

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("RegularSparkApp") \
        .master(args.spark_url) \
        .getOrCreate()

    sc = spark.sparkContext
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)

    total_time = []
    total_RAM = []

    for i in tqdm(range(100)):
        start_time = time.time()

        model_rf = RandomForestRegressionModel.load(args.model_path)
        df = spark.read.csv(args.data_path, header=True)
        df = preprocessing_data(df)
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        df = assembler.transform(df)
        predictions = model_rf.transform(df)

        end_time = time.time()

        total_time.append(end_time - start_time)
        total_RAM.append(get_executor_memory(sc))

    draw_graph(total_time, total_RAM, './not_optim.png')

    print()
    print('Average memory(MB):', np.mean(total_RAM))
    print('Average time(c):', np.mean(total_time))
    spark.stop()