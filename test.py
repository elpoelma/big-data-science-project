import findspark
findspark.init()
import sys
import pyspark

conf = pyspark.SparkConf().setMaster("local[2]").setAppName("loading")
sc = pyspark.SparkContext(conf=conf)
spark = pyspark.sql.SparkSession(sc)

df = spark.read.format("scifio").option("path", sys.argv[1]).option("filelimit", 1).option("imagelimit", 100).option("numpartitionsperfile", 5).option("channels", "1").option("masked", True).load().cache()
print(df)
