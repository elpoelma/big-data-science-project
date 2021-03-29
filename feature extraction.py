import findspark
import pyspark
import numpy as np
import cv2 as cv
import math
findspark.init()

FAST = cv.xfeatures2d.StarDetector_create(10)
BRIEF = cv.xfeatures2d.BriefDescriptorExtractor_create()


def load_data():
    fileLimit = 2
    path = "./data"

    conf = pyspark.SparkConf().setMaster("local[2]").setAppName("loading")
    sc = pyspark.SparkContext(conf=conf)
    spark = pyspark.sql.SparkSession(sc)

    channels = "1,2,3,4,5,6,7,8,9"
    parti = 5
    df = spark.read.format("scifio").option("path", path).option("filelimit", fileLimit).option("numpartitionsperfile",
                                                                                                parti).option(
        "channels", channels).option("masked", True).load().cache()

    # print("OUTPUT BELOW:")
    # print(df.select('width'))
    return df


def calc_descriptor(image, features):
    global FAST
    global BRIEF
    for i, channel in enumerate(image):
        reshaped = cv.convertScaleAbs(channel)
        reshaped = reshaped.astype('uint8')
        kp = FAST.detect(reshaped, None)
        print(kp)
        _, des = BRIEF.compute(reshaped, kp)

        print(len(des))
        features["descriptor"].append(des)


def calc_mean_intensity(image, features):
    features["mean_intensity"] = []
    for i, channel in enumerate(image):
        features["mean_intensity"].append(channel.mean())


def calc_circularity(mask, features):
    features["circularity"] = []
    for i, channel in enumerate(mask):
        features["circularity"].append(4 * np.pi * features["area"][i] / math.pow(features["perimeter"][i], 2))


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


def main():
    df = load_data()
    row = df.take(1)[0]
    mask = np.reshape(row.mask, (9, row.width, row.height))
    channel_data = np.ma.array(np.reshape(row.data, (9, row.width, row.height)), mask=mask)
    for i, channel in enumerate(channel_data):
        # color_channel = cv.cvtColor(channel, cv.COLOR_GRAY2BGR)
        cv.imwrite("images/Channel%d.png" % (i+1), channel)
    features = {}
    calc_area(mask, features)
    calc_perimeter(mask, features)
    calc_circularity(mask, features)
    calc_mean_intensity(channel_data, features)
    calc_descriptor(channel_data, features)

    for key, value in features.items():
        print("%s: %s" % (key, str(value)))


if __name__ == '__main__':
    main()
