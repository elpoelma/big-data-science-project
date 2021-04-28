import os
import pyspark

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml.linalg import Vectors
import cv2 as cv
import csv
from progresstest import progress_bar
import json

def init_spark_session():
    conf = pyspark.SparkConf().setMaster("local[2]").setAppName("loading")
    sc = pyspark.SparkContext(conf=conf)
    spark = pyspark.sql.SparkSession(sc)
    return spark

def load_dataframe(spark_context, path, fileLimit):
    return spark_context\
            .read.format("scifio").option("path", path)\
            .option("filelimit", fileLimit).option("numpartitionsperfile", 5)\
            .option("channels", "1,2,3,4,5,6,7").option("masked", True).load()

def row_to_image(row, channel=1, nr_of_channels=7):
    return np.reshape(row.data, (nr_of_channels, row.width,row.height))[channel].astype('uint8')

def image_to_df(image, spark_context):
    image_vectors = [(Vectors.dense([10*p, index[0], index[1]]), 1.0) for index, p in np.ndenumerate(image)]
    return spark_context.createDataFrame(image_vectors, ["features", "weighCol"])

def row_to_mask(row, channel=1, nr_of_channels=7):
    return np.reshape(row.mask, (nr_of_channels, row.width,row.height))[channel]
    
def apply_opening(mask, shape=(8, 8)):
    mask = mask.astype(np.float32)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,shape)
    opened_mask = cv.morphologyEx(mask,cv.MORPH_CLOSE,kernel)
    return opened_mask


def calculate_performance(evaluation_mask, ground_truth_mask):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i, evaluation_pixel in np.ndenumerate(evaluation_mask):
        ground_truth_pixel = ground_truth_mask[i]
        if ground_truth_pixel and evaluation_pixel:
            TP += 1
        elif ground_truth_pixel and not evaluation_pixel:
            FP += 1
        elif not ground_truth_pixel and not evaluation_pixel:
            TN += 1
        elif not ground_truth_pixel and evaluation_pixel:
            FN += 1
    return (TP, FP, TN, FN)

def balanced_accuracy(evaluation_mask, ground_truth_mask):
    TP, FP, TN, FN = calculate_performance(evaluation_mask, ground_truth_mask)
    tpr, tnr = (1, 1)
    if (TP + FN > 0):
        tpr = TP/(TP + FN)
    if (TN + FP > 0):
        tnr = TN/(TN + FP)
    return (tpr + tnr)/2

def calculate_accuracy(evaluation_mask, ground_truth_mask):
    TP, FP, TN, FN = calculate_performance(evaluation_mask, ground_truth_mask)
    return (TP + TN)/(TP + FP + TN + FN)


        

def canny_masking(image, threshold1, threshold2, opening_shape):
    edges = cv.Canny(image,threshold1=threshold1,threshold2=threshold2)
    mask = apply_opening(edges, shape=opening_shape).astype('bool')
    return mask

def calculate_masks(cell, threshold1, threshold2, opening_shape):
    nr_of_channels = 7
    cell_channels = np.reshape(cell.data, (nr_of_channels, cell.width, cell.height)).astype('uint8')
    cell_masks = np.reshape(cell.mask, (nr_of_channels, cell.width, cell.height))
    predicted_masks = np.array([], dtype='bool')
    accuracies = []
    for i, cell_channel in enumerate(cell_channels):
        ground_truth_mask = cell_masks[i]
        mask = canny_masking(cell_channel, threshold1, threshold2, opening_shape).astype('bool')
        balanced_acc = balanced_accuracy(mask, ground_truth_mask)
        accuracies.append(balanced_acc)
        mask = np.reshape(mask, (cell.width*cell.height))
        predicted_masks = np.concatenate((predicted_masks, mask))
    return predicted_masks, np.array(accuracies)





class CannyEdgeMaskingModel:

    def __init__(self, nr_of_channels) -> None:
        self.nr_of_channels = nr_of_channels
        self.parameters = [(50, 100, (8, 8)) for _ in range(nr_of_channels)]

    def set_parameters(self, channel, threshold1, threshold2, opening_shape):
        if(channel >= 0 and channel < self.nr_of_channels):
            self.parameters[channel] = (threshold1, threshold2, opening_shape)

    def train(self, cells_rdd, threshold1_range, threshold2_range, opening_shape_range):
        best_accuracies = [0 for _ in range(self.nr_of_channels)]
        best_params = [None for _ in range(self.nr_of_channels)]
        progress = 0
        for threshold1 in threshold1_range:
            for threshold2 in threshold2_range:
                for opening_shape in opening_shape_range:
                    progress_bar(progress, len(threshold1_range)*len(threshold2_range)*len(opening_shape_range))
                    accuracy_sums = cells_rdd.map(lambda cell: calculate_masks(cell, threshold1, threshold2, opening_shape))\
                            .map(lambda x: x[1])\
                            .aggregate((np.zeros(self.nr_of_channels), 0),
                                        lambda acc1, acc2: (acc1[0] + acc2, acc1[1] + 1),
                                        lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1]))
                    accuracies = accuracy_sums[0]/accuracy_sums[1]
                    for i, accuracy in enumerate(accuracies):
                        if accuracy > best_accuracies[i]:
                            best_accuracies[i] = accuracy
                            best_params[i] = (threshold1, threshold2, opening_shape)
                    progress += 1
        self.parameters = best_params
        return best_accuracies


    def save_model(self, file):
        data = {}
        for channel in range(self.nr_of_channels):
            params = self.parameters[channel]
            data[channel] = {"threshold1": params[0],
                             "threshold2": params[1],
                             "opening_shape": params[2]}
        with open(file, 'w') as jsonfile:
            json.dump(data, jsonfile)


    @classmethod
    def load_model(cls, file):
        with open(file, 'r') as jsonfile:
            data = json.load(jsonfile)
            nr_of_channels = len(data)
            model = cls(nr_of_channels)
            for channel in data:
                model.set_parameters(int(channel), 
                                     data[channel]["threshold1"],
                                     data[channel]["threshold2"],
                                     data[channel]["opening_shape"])
        return model

    def predict(self):
        pass

if __name__ == "__main__":
    spark = init_spark_session()

    path = '../data/'
    fileLimit = 2

    df = load_dataframe(spark, path, fileLimit)
    df = df.limit(10)
    model = CannyEdgeMaskingModel.load_model("test.json")
    # model = CannyEdgeMaskingModel(7)

    # train_accuracies = model.train(df.rdd, range(10, 111, 10), range(10, 111, 10),[(8, 8)]) 

    print("Parameters: " + str(model.parameters))
    model.save_model("test.json")