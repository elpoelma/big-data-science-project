import os
import pyspark

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml.linalg import Vectors
import cv2 as cv


def init_spark_session():
    conf = pyspark.SparkConf().setMaster("local[2]").setAppName("loading").set('spark.jars', '../scifio-spark-datasource-uber.jar')
    sc = pyspark.SparkContext(conf=conf)
    spark = pyspark.sql.SparkSession(sc)
    return spark

def load_dataframe(spark_context, path, fileLimit):
    return spark_context\
            .read.format("scifio").option("path", path)\
            .option("filelimit", fileLimit).option("numpartitionsperfile", 5)\
            .option("channels", "1,2,3,4,5,6,7").option("masked", True).load()

def row_to_image(row, channel=1, nr_of_channels=7):
    return np.reshape(row.data, (nr_of_channels, row.width,row.height))[channel]

def image_to_df(image, spark_context):
    image_vectors = [(Vectors.dense([10*p, index[0], index[1]]), 1.0) for index, p in np.ndenumerate(image)]
    return spark_context.createDataFrame(image_vectors, ["features", "weighCol"])

def row_to_mask(row, channel=1, nr_of_channels=7):
    return np.reshape(row.mask, (nr_of_channels, row.width,row.height))[channel]
    
def apply_opening(mask):
    mask = mask.astype(np.float32)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(8,8))
    opened_mask = cv.morphologyEx(mask,cv.MORPH_CLOSE,kernel)
    return opened_mask

def calculate_accuracy(evaluation_mask, ground_truth_mask):
    (evaluation_mask == ground_truth_mask).mean()