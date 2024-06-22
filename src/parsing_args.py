import argparse

parser = argparse.ArgumentParser(description='ЗRunning Spark script with parameters')
parser.add_argument('--spark-url', type=str, required=True, help='URL for Spark Master')
parser.add_argument('--model-path', type=str, required=True, help='Path to model in HDFS')
parser.add_argument('--data-path', type=str, required=True, help='Path to CSV file in HDFS')
args = parser.parse_args()