# Assignment 4 Group # 3
# Team members - Sahil Gupta (sahilg1), Yatin Rehani (yrehani)
# Code reference - http://www.techpoweredmath.com/spark-dataframes-mllib-tutorial/

import pyspark.mllib.regression
from pyspark.mllib.regression import LabeledPoint

from numpy import array
from pyspark import SparkContext


import pyspark.mllib
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.feature import StandardScaler

from pyspark.sql import Row
from pyspark.sql.functions import *

#creating the sc object
sc = SparkContext("local", "Group 3 App")



#Reading the file into spark
houses_rdd = sc.textFile('./Boston_house/boston_house.csv')
#Splitting the file by commas
houses_rdd = houses_rdd.map(lambda line: line.split(","))
header = houses_rdd.first()
#Getting header names
houses_rdd_1 = houses_rdd.filter(lambda line: line != header)



df = houses_rdd_1.map(lambda l: Row(CRIM=l[0], ZN=l[1], INDUS=l[2], CHAS=l[3], NOl=l[4], RM=l[5], AGE=l[6], DIS=l[7], RAD=l[8], TAl=l[9], PTRATIO=l[10], B=l[11], LSTAT=l[12], AA_MEDV=l[13]))
features = df.map(lambda row: row[1:])

model = StandardScaler(withMean=True, withStd=True).fit(features)
transformed = model.transform(features)

boston_data = df.map(lambda row: row[0])
#getting the transformed data
transformed_data = boston_data.zip(transformed)

transformed_data = transformed_data.map(lambda row: LabeledPoint(row[0],[row[1]]))



#Applying linear regression
linear_model = LinearRegressionWithSGD.train(transformed_data, intercept=True)


###########Testing the model using verification.csv

verify_houses_data = sc.textFile('./Boston_house/verification.csv')

#splitting the line by commas
verify_houses_data = verify_houses_data.map(lambda line: line.split(","))
header = verify_houses_data.first()
header_data = verify_houses_data.filter(lambda line: line != header)
verifying_df = header_data.map(lambda l: Row(CRIM=l[0], ZN=l[1], INDUS=l[2], CHAS=l[3], NOl=l[4], RM=l[5], AGE=l[6], DIS=l[7], RAD=l[8], TAl=l[9], PTRATIO=l[10], B=l[11], LSTAT=l[12]))
features = verifying_df.map(lambda row: row[0:])
features_transform = model.transform(features)

#predicting the values
predicted_result= linear_model.predict(features_transform)
#writing out the result as a text file
predicted_result.saveAsTextFile('./Boston_house/predicted_results');
