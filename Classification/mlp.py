from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import *

from pyspark.mllib.util import MLUtils

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import Normalizer
from pyspark.sql.functions import *

# This filepath will vary based on your machine...
HOME_PATH = "/home/michaelnguyen/Documents/Fall2017/BIG_DATA/BigData_ADNI_project/Bayes/"
# FILE_PATH = HOME_PATH + "ADNIMERGE_FILTERED_NOHEAD.csv"
# FILE_PATH = HOME_PATH + "GENETIC_FILTERED_NOHEAD.csv"
FILE_PATH = HOME_PATH + "IMAGING_FILTERED_NOHEAD.csv"
# FILE_PATH = HOME_PATH + "MERGED_FILTERED_NOHEAD.csv"

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)
sc.setLogLevel("WARN")

raw_data = spark.read.text(FILE_PATH).rdd

# Select the data
parts = raw_data.map(lambda row: row.value.split(","))
labeledFeatures = parts.map(lambda p: ( int(p[0]), Vectors.dense([
	float(p[x]) for x in range( 1, len(p) )
	]) ) )

data = spark.createDataFrame(labeledFeatures, ["label", "features"])

data = spark.createDataFrame(labeledFeatures, ["label", "features"])

print( data.show() )

splits = data.randomSplit([0.7, 0.3])
trainset = splits[0]
testset = splits[1]

#Input layer (8 features) and output layer (2 classes)
# layers = [66, 128, 512, 256, 100, 2]
# layers = [72, 128, 512, 256, 100, 2]
layers = [14, 128, 512, 256, 100, 2]

trainer = MultilayerPerceptronClassifier(maxIter=10000, layers=layers, blockSize=128)
model = trainer.fit( trainset )

result = model.transform(testset)
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Some examples:")
result.show(n = 20)
