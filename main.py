import findspark
findspark.init()

import pyspark
from pyspark.sql.functions import avg
import sys


def load_data():
    path = sys.argv[1]
    fileLimit = 2

    conf = pyspark.SparkConf().setMaster("local[2]").setAppName("loading")
    sc = pyspark.SparkContext(conf=conf)
    spark = pyspark.sql.SparkSession(sc)

    channels = "1,2,3,4,5,6,7,8,9"
    parti = 5
    df = spark.read.format("scifio").option("path", path).option("filelimit", fileLimit).option("numpartitionsperfile", parti).option("channels", channels).option("masked", True).load().cache()

    print("OUTPUT BELOW:")
    # df.show()
    # print(df.count())
    return df


print(load_data().count())
