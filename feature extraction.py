import findspark
import pyspark
import numpy as np
import cv2 as cv
import math
from outlier_detection import OutlierModel
findspark.init()

surf = cv.xfeatures2d_SURF.create(hessianThreshold=400, upright=True)


def load_data(spark):
    fileLimit = 2
    path = "./data"

    channels = "1,2,3,4,5,6,7,8,9"
    parti = 5
    df = spark.read.format("scifio").option("path", path).option("filelimit", fileLimit).option("numpartitionsperfile",
                                                                                                parti).option(
        "channels", channels).option("masked", True).load().cache()

    # print("OUTPUT BELOW:")
    # print(df.select('width'))
    return df


def calc_descriptor(image, features):
    for i, channel in enumerate(image):
        reshaped = cv.convertScaleAbs(channel)
        reshaped = reshaped.astype('uint8')
        _, des = surf.detectAndCompute(reshaped, None)
        print(des)

        print(len(des))
        features["descriptor"].append(des)


def calc_mean_intensity(image, features):
    features["mean_intensity"] = []
    for i, channel in enumerate(image):
        features["mean_intensity"].append(channel.mean())


def calc_circularity(mask, features):
    features["circularity"] = []
    for i, channel in enumerate(mask):
        features["circularity"].append(
            4 * np.pi * features["area"][i] / math.pow(features["perimeter"][i], 2)
            if features["perimeter"][i] > 0
            else 0
        )


def neighbour_count(mat, x, y):
    count = 0
    if x > 0 and not mat[x - 1][y]:
        count += 1
    if x < len(mat) - 1 and not mat[x + 1][y]:
        count += 1
    if y > 0 and not mat[x][y - 1]:
        count += 1
    if y < len(mat[x]) - 1 and not mat[x][y + 1]:
        count += 1

    return count


def calc_perimeter(mask, features):
    features["perimeter"] = []

    for channel in mask:  # iterate channels
        perimeter = 0
        for x, row in enumerate(channel):
            for y, v in enumerate(row):
                if v:
                    perimeter += neighbour_count(channel, x, y)
        features["perimeter"].append(perimeter)


def calc_area(mask, features):
    features["area"] = []
    for channel in mask:
        features["area"].append(np.count_nonzero(channel))


def calculate_features(row):
    mask = np.reshape(row.mask, (9, row.width, row.height))
    channel_data = np.ma.array(np.reshape(row.data, (9, row.width, row.height)), mask=mask)
    for i, channel in enumerate(channel_data):
        # color_channel = cv.cvtColor(channel, cv.COLOR_GRAY2BGR)
        cv.imwrite("images/Channel%d.png" % (i + 1), channel)
    features = {}
    calc_area(mask, features)
    calc_perimeter(mask, features)
    calc_circularity(mask, features)
    calc_mean_intensity(channel_data, features)

    return [(k, v) for k, v in features.items()]


def main():
    conf = pyspark.SparkConf().setMaster("local[2]").setAppName("loading")
    sc = pyspark.SparkContext(conf=conf)
    spark = pyspark.sql.SparkSession(sc)
    spark.sparkContext.setLogLevel('WARN')
    model = OutlierModel()

    df = load_data(spark)

    # training
    # df = df.rdd.flatMap(calculate_features)
    # model.train(df)

    model.read("outlier_model.json")
    df = df.rdd.map(calculate_features)
    print(f"count before: {df.count()}")
    df = df.filter(model.is_no_outlier)
    print(f"count after: {df.count()}")
    # for datum in df.rdd.toLocalIterator():
    #     row = datum[0]

    # calc_descriptor(channel_data, features)

        # model.add_sample(features)
    # model.write()

    # for key, value in features.items():
    #     print("%s: %s" % (key, str(value)))


if __name__ == '__main__':
    main()
