import findspark
findspark.init()

import pyspark
from pyspark.sql.functions import avg

fileLimit = 2
path = "./data"

conf = pyspark.SparkConf().setMaster("local[2]").setAppName("loading").set('spark.jars', './spark-scifio/target/scifio-spark-datasource-fat.jar')
sc = pyspark.SparkContext(conf=conf)
spark = pyspark.sql.SparkSession(sc)

df = spark.read.format("scifio").option("path", path).option("filelimit", fileLimit).option("numpartitionsperfile", 5).option("channels", "1").option("masked", True).load().cache()

print("OUTPUT BELOW:")
df.agg(avg(df["width"])).show()
