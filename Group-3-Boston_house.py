# Assignment 4 Group # 3
# Team members - Sahil Gupta (sahilg1), Yatin Rehani (yrehani)
# Code reference - http://www.techpoweredmath.com/spark-dataframes-mllib-tutorial/

import pyspark.mllib
import pyspark.mllib.regression
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import *
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.feature import StandardScaler, StandardScalerModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.util import MLUtils

houses_rdd = sc.textFile('./tmp/boston-data/boston_house.csv')
houses_rdd = houses_rdd.map(lambda line: line.split(\",\"))

header = houses_rdd.first() #we want all members of the RDD that are NOT equal to the header
houses_rdd = houses_rdd.filter(lambda line:line != header)

df = houses_rdd.map(lambda line: Row(CRIM = line[0], ZN = line[1], INDUS=line[2], CHAS=line[3], NOX=line[4], RM=line[5], AGE=line[6], DIS=line[7], RAD=line[8], TAX=line[9], PTRATIO=line[10], B=line[11], LSTAT=line[12])).toDF()
features = df.map(lambda row: row[1:])
model = StandardScaler(withMean=True, withStd=True).fit(features)

transformedFeatures = model.transform(features)
boston_data = df.map(lambda row:row[len(row)-1])

transformedData = boston_data.zip(transformedFeatures)
transformedData = transformedData.map(lambda row: LabeledPoint(row[0],[row[1]]))

linearModel = LinearRegressionWithSGD.train(transformedData, intercept=True)

verifyingHousesData = sc.textFile('./tmp/boston-data/verification.csv')\n
verifyingHousesData = verifyingHousesData.map(lambda line: line.split(\",\"))

verifyingHeaders = verifyingHousesData.first()

verifyingHousesData = verifyingHousesData.filter(lambda line: line != verifyingHeaders)
verifying_df = verifyingHousesData.map(lambda line: Row(CRIM = line[0], ZN=line[1],INDUS=line[2],CHAS=line[3],NOX=line[4],RM=line[5],AGE=line[6],DIS=line[7],RAD=line[8],TAX=line[9],PTRATIO=line[10],B=line[11],LSTAT=line[12])).toDF()\n\n
testing_features = verifying_df.map(lambda row: row[0:])
testingTransformedFeatures = model.transform(testing_features)
predicted_result = linearModel.predict(testingTransformedFeatures)

predicted_result.saveAsTextFile('./tmp/boston-data/predicted_resuts')