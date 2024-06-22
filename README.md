# Big_Data_Hadoop_Spark
The project is a data processing and storage system using Hadoop and Spark. The main goal of the project is to provide a distributed, scalable, reliable and efficient solution for data processing and storage.

## Project structure
    .
    ├── 1 DataNode 
    |   ├── docker-compose.yml       # Docker Compose configuration for a setup with one DataNode.
    |   ├── dockerfile               # Dockerfile for setting up the spark-worker.
    |   ├── hadoop.env               # Environment file containing Hadoop configuration variables.
    |   ├── not_optim.png            # Image showing performance or results without optimizations.
    |   └── optim.png                # Image showing performance or results with optimizations.  
    ├── 3 DataNode
    |   ├── docker-compose.yml       # Docker Compose configuration for a setup with three DataNodes.
    |   ├── dockerfile               # Dockerfile for setting up the spark-worker.
    |   ├── hadoop.env               # Environment file containing Hadoop configuration variables.
    |   ├── not_optim.png            # Image showing performance or results without optimizations.
    |   └── optim.png                # Image showing performance or results with optimizations. 
    ├── data
    |   ├── random_forest_model      # Directory containing the saved RandomForest model.
    |   └── FinalDataset.csv         # CSV file with the dataset used for predictions.
    ├── notebooks
    |   └── model_train.ipynb        # Jupyter Notebook for training the RandomForest model.
    ├── src   
    |   ├── functions.py             # Script containing helper functions used across other scripts.
    |   ├── optimSpark.py            # Script for executing optimized computations using Spark.
    |   ├── parsing_args.py          # Script for parsing command-line arguments and executing tasks based on them.
    |   └── regularSpark.py          # Script for executing basic computations using Spark without optimizations.
    └── README.md                  
## Dataset
The final dataset was created based on data provided by <a href="https://www.kaggle.com/datasets/nfedorov/mts-ml-cookies/data?select=dataset_full.feather"> MTS </a>, with the main purpose of predicting the age of users. The final dataset contains only a part of the original data set, and preliminary processing has been carried out aimed at removing duplicate records about the same user. As a result of these steps, a final dataset was formed, ready for further use in analysis and machine learning tasks.
## Run project
### 1DataNode
- Container startup
```bash
docker-compose up --build
```
- Uploading data to HDFS
```bash
# dataset and model are on hdfs

# containerize
docker cp FinalDataset.csv namenode:/
docker cp random_forest_model namenode:/

# connect to the desired container
docker exec -it namenode /bin/bash

# uploading data to hdfs
hdfs dfs -put FinalDataset.csv /
hdfs dfs -put random_forest_model /
```
- Start the execution of the file regularSpark.py
```bash
docker exec -it spark-worker-1 /spark/bin/spark-submit /opt/spark-apps/regularSpark.py --spark-url spark://spark-master:7077 --model-path hdfs://namenode:9001/random_forest_model --data-path hdfs://namenode:9001/FinalDataset.csv
```
- Start the execution of the file optimSpark.py
```bash
docker exec -it spark-worker-1 /spark/bin/spark-submit /opt/spark-apps/optimSpark.py --spark-url spark://spark-master:7077 --model-path hdfs://namenode:9001/random_forest_model --data-path hdfs://namenode:9001/FinalDataset.csv
```
- Unload graphs
```bash
docker cp <id container>:optim.png ./
docker cp <id container>:not_optim.png ./
```
### 3DataNode
- Container startup
```bash
docker-compose up --build
```
- Uploading data to HDFS
```bash
# dataset and model are on hdfs

# containerize
docker cp FinalDataset.csv namenode2:/
docker cp random_forest_model namenode2:/

# connect to the desired container
docker exec -it namenode2 /bin/bash

# uploading data to hdfs
hdfs dfs -put FinalDataset.csv /
hdfs dfs -put random_forest_model /
```
- Start the execution of the file regularSpark.py
```bash
docker exec -it spark-worker-2 /spark/bin/spark-submit /opt/spark-apps/regularSpark.py --spark-url spark://spark-master2:7077 --model-path hdfs://namenode2:9001/random_forest_model --data-path hdfs://namenode2:9001/FinalDataset.csv
```
- Start the execution of the file optimSpark.py
```bash
docker exec -it spark-worker-2 /spark/bin/spark-submit /opt/spark-apps/optimSpark.py --spark-url spark://spark-master2:7077 --model-path hdfs://namenode2:9001/random_forest_model --data-path hdfs://namenode2:9001/FinalDataset.csv
```
- Unload graphs
```bash
docker cp <id container>:optim.png ./
docker cp <id container>:not_optim.png ./
```
## Demonstration results
### 1DataNode
#### Regular realization
![not_optim](https://github.com/o-frolova/Big_Data_Hadoop_Spark/assets/128040555/5e38b6f7-6a45-436a-b89e-408c5f6bb717)


#### Optimization realization
![optim](https://github.com/o-frolova/Big_Data_Hadoop_Spark/assets/128040555/f0dc2a94-ec15-4763-b2f6-20a478165f3f)


### 3DataNode
#### Regular realization
![not_optim](https://github.com/o-frolova/Big_Data_Hadoop_Spark/assets/128040555/2960d9b1-c52b-4c6b-b9a5-b9dcd04b6264)


#### Optimization realization
![optim](https://github.com/o-frolova/Big_Data_Hadoop_Spark/assets/128040555/aca43ca5-4fc2-45de-8bfb-12ed087eabc8)

