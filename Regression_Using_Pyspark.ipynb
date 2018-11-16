#Regression Using Pyspark

#Pyspark Initializasing
# to make pyspark importable as a regular library
import findspark
findspark.init()

import pyspark

from pyspark import SparkContext
sc = SparkContext.getOrCreate()

#initializasing SparkSession for creating Spark DataFrame
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()


#Load Libraries
# Data Frame spark profiling 
from pyspark.sql.types import IntegerType, StringType, DoubleType, ShortType, DecimalType
import pyspark.sql.functions as func
from pyspark.sql.functions import isnull
from pyspark.sql.functions import isnan, when, count, col
from pyspark.sql.functions import mean
from pyspark.sql.functions import round
from pyspark.sql.types import Row
import matplotlib.pyplot as plt
from pyspark.sql.functions import udf

# Pandas DF operation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import array

# Modeling + Evaluation
from pyspark.ml.feature import VectorAssembler, VectorIndexer, OneHotEncoder, StringIndexer
from pyspark.sql.functions import when
from pyspark.sql import functions as F
from pyspark.sql.functions import avg
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder 
from sklearn.metrics import log_loss
from pyspark.sql.functions import corr
import pyspark.sql.functions as fn 
from pyspark.sql.functions import rank,sum,col
from pyspark.sql import Window

window = Window.rowsBetween(Window.unboundedPreceding,Window.unboundedFollowing)


#Load Data to Spark DataFrame
#Initializing File Type and path for data train
file_type = 'text'
path=r'train_house_pricing.csv'
delimeter=','

#Function load data
def load_data(file_type):
    """input type of file "text" or "parquet" and Return pyspark dataframe"""
    if file_type =="text": # use text as file type input
        df = spark.read.option("header", "true") \
                       .option("delimeter",delimeter)\
                       .option("inferSchema", "true") \
                       .csv(path)  #path file that you want import
    else:  
        df= spark.read.parquet("example.parquet") #path file that you want import
    return df
    
#call function load_data
df = load_data(file_type)

#Initializing File Type and path for data test
file_type = 'text'
path=r'test.csv'
delimeter=','

#call function load_data
test_data = load_data(file_type)



#Check Data
#check type of data train and data test

type(df)
type(test_data)

#show 5 observation in data train
df.show(5)

#show 5 observation in data test
test_data.show(5)

#Print Schema and count number of columns from data train
len(df.columns), df.printSchema()

#Print Schema and count number of columns from data test
len(test_data.columns), test_data.printSchema()

#rename Target('SalePrice') to 'label
df_final = df.withColumnRenamed('SalePrice','label')

#Change data types in data train
df_final=df_final.withColumn("LotFrontage", df_final["LotFrontage"].cast(IntegerType()))
df_final=df_final.withColumn("OverallQual", df_final["OverallQual"].cast(StringType()))
df_final=df_final.withColumn("OverallCond", df_final["OverallCond"].cast(StringType()))
df_final=df_final.withColumn("MasVnrArea", df_final["MasVnrArea"].cast(IntegerType()))
df_final=df_final.withColumn("GarageYrBlt", df_final["GarageYrBlt"].cast(IntegerType()))

#Change data types in data test
test_data=test_data.withColumn("LotFrontage", test_data["LotFrontage"].cast(IntegerType()))
test_data=test_data.withColumn("OverallQual", test_data["OverallQual"].cast(StringType()))
test_data=test_data.withColumn("OverallCond", test_data["OverallCond"].cast(StringType()))
test_data=test_data.withColumn("MasVnrArea", test_data["MasVnrArea"].cast(IntegerType()))
test_data=test_data.withColumn("GarageYrBlt", test_data["GarageYrBlt"].cast(IntegerType()))
test_data=test_data.withColumn("BsmtFinSF2", test_data["BsmtFinSF2"].cast(IntegerType()))
test_data=test_data.withColumn("BsmtFinSF1", test_data["BsmtFinSF1"].cast(IntegerType()))
test_data=test_data.withColumn("TotalBsmtSF", test_data["TotalBsmtSF"].cast(IntegerType()))
test_data=test_data.withColumn("BsmtUnfSF", test_data["BsmtUnfSF"].cast(IntegerType()))
test_data=test_data.withColumn("BsmtFullBath", test_data["BsmtFullBath"].cast(IntegerType()))
test_data=test_data.withColumn("BsmtHalfBath", test_data["BsmtHalfBath"].cast(IntegerType()))
test_data=test_data.withColumn("GarageArea", test_data["GarageArea"].cast(IntegerType()))
test_data=test_data.withColumn("GarageCars", test_data["GarageCars"].cast(IntegerType()))

#count number of observation in data train
df_final.count()


#Define categorical and nummerical variable in df_final (data train)
#Categorical and numerical variable
#just will select string data type
cat_cols = [item[0] for item in df_final.dtypes if item[1].startswith('string')] 
print("cat_cols:", cat_cols)

#just will select integer or double data type
num_cols = [item[0] for item in df_final.dtypes if item[1].startswith('int') | item[1].startswith('double')] 
print("num_cols:", num_cols)

#Save column Id
num_id=num_cols.pop(0)
print("num_id:", num_id)
num_id=[num_id]
print(num_id)

#Remove column 'label' from numerical columns group
num_cols.remove('label') #label is removed because it's the target to validate the model
print("num_cols:", num_cols)


#Define categorical and nummerical variable in test_data (data test)
#Categorical and numerical variable
#just will select string data type
cat_cols_test = [item[0] for item in test_data.dtypes if item[1].startswith('string')] 
print("cat_cols_test:", cat_cols_test)

#just will select integer or double data type
num_cols_test = [item[0] for item in test_data.dtypes if item[1].startswith('int') | item[1].startswith('double')] 
print("num_cols_test:", num_cols_test)

#Save column Id
num_id_test=num_cols_test.pop(0)
print("num_id_test:", num_id_test)
num_id_test=[num_id_test]
print(num_id_test)
print(num_cols_test)

#count observation in data test
test_data.count()

#count number of numerical and categorical columns in data test
len(num_cols_test), len(cat_cols_test)


#Check Missing Value in data train
#Check Missing Value in Pyspark Dataframe
def count_nulls(df_final):
    """Input pyspark dataframe and return list of columns with missing value and it's total value"""
    null_counts = []          #make an empty list to hold our results
    for col in df_final.dtypes:     #iterate through the column data types we saw above, e.g. ('C0', 'bigint')
        cname = col[0]        #splits out the column name, e.g. 'C0'    
        ctype = col[1]        #splits out the column type, e.g. 'bigint'
        nulls = df_final.where( df_final[cname].isNull() ).count() #check count of null in column name
        result = tuple([cname, nulls])  #new tuple, (column name, null count)
        null_counts.append(result)      #put the new tuple in our result list
    null_counts=[(x,y) for (x,y) in null_counts if y!=0]  #view just columns that have missing values
    return null_counts
    
    
#call function check missing values
null_counts = count_nulls(df_final)
null_counts

#From null_counts, we just take information of columns name and save in list "list_cols_miss", like in the script below:
list_cols_miss=[x[0] for x in null_counts]
list_cols_miss

#Create dataframe which just has list_cols_miss
df_miss= df_final.select(*list_cols_miss)

#view data types in df_miss
df_miss.dtypes

#Define categorical columns and numerical columns which have missing value.
### for categorical columns
catcolums_miss=[item[0] for item in df_miss.dtypes if item[1].startswith('string')]  #will select name of column with string data type
print("catcolums_miss:", catcolums_miss)

### for numerical columns
numcolumns_miss = [item[0] for item in df_miss.dtypes if item[1].startswith('int') | item[1].startswith('double')] #will select name of column with integer or double data type
print("numcolumns_miss:", numcolumns_miss)

#fill missing value in numerical variable with average
for i in numcolumns_miss:
    meanvalue = df_final.select(round(mean(i))).collect()[0][0] #calculate average in each numerical column
    print(i, meanvalue) #print name of columns and it's average value
    df_final=df_final.na.fill({i:meanvalue}) #fill missing value in each columns with it's average value
    
#Check Missing value after filling
null_counts = count_nulls(df_final)
null_counts

#Drop missing value
df_Nomiss=df_final.na.drop()

#fill missing value in categorical variable with most frequent
for x in catcolums_miss:
    mode=df_Nomiss.groupBy(x).count().sort(col("count").desc()).collect()[0][0] #group by based on categories and count each categories and sort descending then take the first value in column
    print(x, mode) #print name of columns and it's most categories 
    df_final = df_final.na.fill({x:mode}) #fill missing value in each columns with most frequent
    
#Check Missing Value in data test
#We will cleansing missing values in pyspark dataframe.
#Call function to count missing values in test_data
null_test= count_nulls(test_data)
null_test

#take just name of columns that have missing values
list_miss_test=[x[0] for x in null_test]
list_miss_test

#Create dataframe which just has list_cols_miss
test_miss= test_data.select(*list_miss_test)

#view data types in df_miss
test_miss.dtypes

#Define categorical columns and numerical columns which have missing value.
### for categorical columns
catcolums_miss_test=[item[0] for item in test_miss.dtypes if item[1].startswith('string')]  #will select name of column with string data type
print("catcolums_miss_test:", catcolums_miss_test)

### for numerical columns
numcolumns_miss_test = [item[0] for item in test_miss.dtypes if item[1].startswith('int') | item[1].startswith('double')] #will select name of column with integer or double data type
print("numcolumns_miss_test:", numcolumns_miss_test)

#fill missing value in numerical variable with average
for i in numcolumns_miss_test:
    meanvalue_test = test_data.select(round(mean(i))).collect()[0][0] #calculate average in each numerical column
    print(i, meanvalue_test) #print name of columns and it's average value
    test_data=test_data.na.fill({i:meanvalue_test}) #fill missing value in each columns with it's average value
    
#Check Missing value after filling
null_test = count_nulls(test_data)
null_test


#Compare categorical columns in df_final and test_data
#Function to check categorical columns in both data train and data test
def check_category2(a1,a2,y):
    """input are two dataframe you want to compare categorical variables and the colomn category name"""
    print('column:',y)
    var1=a1.select([y]).distinct() #define distinct category in column in dataframe1
    var2=a2.select([y]).distinct() #define distinct category in column in dataframe2
    diff2=var2.subtract(var1).collect() #define the different category in dataframe2, return is list
    diff2=[r[y] for r in diff2] #just take the values
    diff1=var1.subtract(var2).collect() #define the different category in dataframe1, return is list
    diff1=[r[y] for r in diff1] #just take the values
    if diff1 == diff2:
        print('diff2:', diff2)
        print('diff1:', diff1)
        print('Columns match!!')
    else:
        if len(diff1)!=0 and len(diff2)==len(diff1):
            print('diff2:', diff2)
            print('diff1:', diff1)
            a2=a2.replace(diff2, diff1, y) #replace the different category in dataframe2 with category in dataframe1
            print('Columns match now!!')
        else:
            if len(diff2)!=len(diff1) and len(diff2)!=0:
                print('diff2:', diff2)
                print('diff1:', diff1)
                dominant1=a1.groupBy(y).count().sort(col("count").desc()).collect()[0][0]
                dominant2=a2.groupBy(y).count().sort(col("count").desc()).collect()[0][0] #define category dominant in dataframe2
                print('dominant2:', dominant2)
                print('dominant1:', dominant1)
                a2=a2.replace(diff2, dominant1, y) #replace different category in dataframe2 with dominant category
                print('Columns match now!!')
            else:     
                print('diff1:', diff1)
                print('diff2:', diff2)
    return a2
    
#call function to check catgories in data train and test, whether same or not, if not, the different categories will be replaced.
for y in cat_cols_test:
    test_data=check_category2(df_final,test_data,y)
 
 
#EDA
#Check distribution in each variables
#Pyspark dataframe has limitation in visualization. Then to create visualization we have to convert pyspark dataframe to pandas dataframe.
# convert spark dataframe to pandas for visualization
df_pd2=df_final.toPandas()

#Barchart for categorical variable
plt.figure(figsize=(20,10))
plt.subplot(221)
sns.countplot(x='MSZoning', data=df_pd2, order=df_pd['MSZoning'].value_counts().index)
plt.title('MSZoning', fontsize=15)
plt.show()

#Barchart for categorical variable
plt.figure(figsize=(20,10))
plt.subplot(221)
sns.countplot(x='Street', data=df_pd2, order=df_pd['Street'].value_counts().index)
plt.title('Street', fontsize=15)
plt.show()

#Barchart for categorical variable
plt.figure(figsize=(20,7))
plt.subplot(131)
sns.countplot(x='LotShape', data=df_pd2, order=df_pd['LotShape'].value_counts().index)
plt.title('LotShape', fontsize=15)
plt.subplot(132)
sns.countplot(x='LandContour', data=df_pd2, order=df_pd['LandContour'].value_counts().index)
plt.title('LandContour', fontsize=15)
plt.subplot(133)
sns.countplot(x='Utilities', data=df_pd2, order=df_pd['Utilities'].value_counts().index)
plt.title('Utilities', fontsize=15)
plt.show()

#distribusi of LotFrontage, LotArea and YearBuilt
plt.figure(figsize=(24,8))
plt.subplot(131)
sns.distplot(df_pd2['LotFrontage'])
plt.subplot(132)
sns.distplot(df_pd2['LotArea'])
plt.subplot(133)
sns.distplot(df_pd2['YearBuilt'])
plt.show()

#distribusi of BsmtFinSF1, MasVnrArea and YearRemodAdd
plt.figure(figsize=(24,8))
plt.subplot(131)
sns.distplot(df_pd2['YearRemodAdd'])
plt.subplot(132)
sns.distplot(df_pd2['MasVnrArea'])
plt.subplot(133)
sns.distplot(df_pd2['BsmtFinSF1'])
plt.show()

#Insignificant Categories in Data train
#Define the threshold for insignificant categories
threshold=97
threshold2=0.7

def replace_cat2(f,cols):
    """input are dataframe and categorical variables, replace insignificant categories (percentage <=0.7) with largest number
    of catgories and output is new dataframe """
    df_percent=f.groupBy(cols).count().sort(col("count").desc())\
                .withColumn('total',sum(col('count')).over(window))\
                .withColumn('Percent',col('count')*100/col('total')) #calculate the percentage-save in Percent columns from each categories
    dominant_cat=df_percent.select(df_percent['Percent']).collect()[0][0] #calculate the highest percentage of category
    count_dist=f.select([cols]).distinct().count() #calculate distinct values in that columns
    if count_dist > 2 and dominant_cat <= threshold :
        print('column:', cols)
        cols_names.append(cols)  #combine with previous list
        replacement=f.groupBy(cols).count().sort(col("count").desc()).collect()[0][0] #define dominant category 
        print("replacement:",replacement)
        replacing.append(replacement) #combine with previous list
        insign_cat=df_percent.filter(df_percent['Percent']< threshold2).select(df_percent[cols]).collect() #calculate insignificant categories
        insign_cat=[r[cols] for r in insign_cat] #just take the values
        category.append(insign_cat) #combine with previous list
        print("insign_cat:",insign_cat)
        f=f.replace(insign_cat,replacement, cols) #replace insignificant categories with dominant categories
    return f
    
#call function replacing insignificant categories 
replacing=[]
cols_names=[]
category=[]
for cols in cat_cols:
    df_final=replace_cat2(df_final,cols)
    
#check length in list cols_names, category and replacing
len(cols_names), len(category), len(replacing)

#Create dataframe of replaced categories
g=spark.createDataFrame(list(zip(cols_names, replacing, category)),['cols_names', 'replacing', 'category'])
g.show(9)

#Replacing Insignificant Categories in data test
#We already have a dataframe containing any categories that need to be replaced, 
#we got it when the process of replacing the insignificant categories in the data train, the data frame is called g. 
#Based on those information, insignificant categories on data test will be replaced.
cols_names_list=g.select('cols_names').collect() #select just cols_names from dataframe g
cols_names_list=[r['cols_names'] for r in cols_names_list] #take just the values

#function to replace insignificant categories in data test
for z in cols_names_list:
    print('cols_names:',z)
    replacement_cat=g.filter(g['cols_names']== z).select(g['replacing']).collect()[0][0] #select values of replacing columns accoring to z in cols_names 
    print('replacement_cat:', replacement_cat)
    insignificant_cat=g.filter(g['cols_names']== z).select(g['category']).collect()[0][0] #select values of category columns accoring to z in cols_names
    print('insignificant_cat:',insignificant_cat)
    test_data=test_data.replace(insignificant_cat,replacement_cat, z) #replace insignificant cat with replacement value


#Handle of outlier in data train
#Calculate Upper&Lower side in pandas dataframe
df_describe=df_pd2.describe()
df_describe

#create quantile dataframe
def quantile(e):
    """Input is dataframe and return new dataframe with value of quantile from numerical columns"""
    percentiles = [0.25, 0.5, 0.75]
    quant=spark.createDataFrame(zip(percentiles, *e.approxQuantile(num_cols, percentiles, 0.0)),
                               ['percentile']+num_cols) #calculate quantile from pyspark dataframe, 0.0 is relativeError,
                                                        #The relative target precision to achieve (>= 0). If set to zero, 
                                                        #the exact quantiles are computed, which could be very expensive
                                                        #and aggregate the result with percentiles variable, 
                                                        #then create pyspark dataframe
    return quant
    
#call function quantile
quantile=quantile(df_final)

#function to calculate uppler side
def upper_value(b,c):
    """Input is quantile dataframe and name of numerical column and Retrun upper value from the column"""
    q1 = b.select(c).collect()[0][0] #select value of q1 from the column
    q2 = b.select(c).collect()[1][0] #select value of q2 from the column
    q3 = b.select(c).collect()[2][0] #select value of q3 from the column
    IQR=q3-q1  #calculate the value of IQR
    upper= q3 + (IQR*1.5)   #calculate the value of upper side
    return upper
    
#function to calculate lower side
def lower_value(b,c):
    """Input is quantile dataframe and name of numerical column and Retrun lower value from the column"""
    q1 = b.select(c).collect()[0][0] #select value of q1 from the column
    q2 = b.select(c).collect()[1][0] #select value of q2 from the column
    q3 = b.select(c).collect()[2][0] #select value of q3 from the column
    IQR=q3-q1                   #calculate the value of IQR
    lower= q1 - (IQR*1.5)       #calculate the value of lower side
    return lower
    
 #function for replacing outlier by upper side
 def replce_outlier_up2(d,col, value):
    """Input is name of numerical column and it's upper side value"""
    #global d
    d=d.withColumn(col, F.when(d[col] > value , value).otherwise(d[col]))
    return d
    
#function for replacing outlier by lower side
def replce_outlier_low2(d,col, value):
    """Input is name of numerical column and it's lower side value"""
    #global df_final
    d=d.withColumn(col, F.when(d[col] < value , value).otherwise(d[col]))
    return d
    
#call function to calculate lower side and replace value under lower side with value lower side
for i in num_cols:
    lower=lower_value(quantile,i)
    df_final=replce_outlier_low2(df_final, i, lower)
    
#call function to calculate upper side and replace value above upper side with value upper side
for x in num_cols:
    upper=upper_value(quantile,x)
    df_final=replce_outlier_up2(df_final, x, upper)
    
#Handle of outlier in data test
#create quantile dataframe
def quantile(e):
    percentiles = [0.25, 0.5, 0.75]
    quant=spark.createDataFrame(zip(percentiles, *e.approxQuantile(num_cols_test, percentiles, 0.0)),
                               ['percentile']+num_cols_test) #calculate quantile from pyspark dataframe, 0.0 is relativeError,
                                                        #The relative target precision to achieve (>= 0). If set to zero, 
                                                        #the exact quantiles are computed, which could be very expensive
                                                        #and aggregate the result with percentiles variable, 
                                                        #then create pyspark dataframe
    return quant
    
#call funtion quantile
quantile=quantile(test_data)

#call function to calculate lower side and replace value under lower side with value lower side
for i in num_cols_test:
    lower=lower_value(quantile,i)
    test_data=replce_outlier_low2(test_data, i, lower)
    
#call function to calculate upper side and replace value above upper side with value upper side
for x in num_cols_test:
    upper=upper_value(quantile,x)
    test_data=replce_outlier_up2(test_data, x, upper)
    
#Feature Engineering
#function to check distinct categories in data train and data test
def check_distinct(a1,a2):
    """input are two dataframe that you want to compare categorical variables and the output is 
    total distinct categories in both dataframe"""
    total1=0
    total2=0
    for y in cat_cols:
        distinct1=a1.select([y]).distinct().count() #count distinct column in dataframe1
        distinct2=a2.select([y]).distinct().count() #count distinct column in dataframe2
        var1=a1.select([y]).distinct().collect() #define distinct category in column in dataframe1
        var1=[r[y] for r in var1]
        var2=a2.select([y]).distinct().collect()
        var2=[r[y] for r in var2]
        total1=total1+distinct1
        total2=total2+distinct2   
    return total1, total2  
    
#function to execute feature engineering
def feature_engineering(a1):
    """Function for feature engineering (StringIndexer and OneHotEncoder process)"""
    cat_columns_string_vec = []
    for c in cat_cols:
        cat_columns_string= c+"_vec"
        cat_columns_string_vec.append(cat_columns_string)
    stringIndexer = [StringIndexer(inputCol=x, outputCol=x+"_Index")
                  for x in cat_cols]
    #use oneHotEncoder to convert categorical variable to binary
    encoder = [OneHotEncoder(inputCol=x+"_Index", outputCol=y)
           for x,y in zip(cat_cols, cat_columns_string_vec)]
    #create list of stringIndexer and encoder with 2 dimension
    tmp = [[i,j] for i,j in zip(stringIndexer, encoder)]
    tmp = [i for sublist in tmp for i in sublist]
    cols_assember=num_id + num_cols + cat_columns_string_vec
    assembler=VectorAssembler(inputCols=cols_assember, outputCol='features')
    tmp += [assembler]
    pipeline=Pipeline(stages=tmp)
    df_final_feat=pipeline.fit(a1).transform(a1)
    return df_final_feat
    
#fucntion to call fucntion feature_engineering and check_distinct
def Main_feature_engineering(df,df2):   
    """Function for calling check_distinct and feature_engineering. Then Join data train and data test if distinct categories 
    between data train and data test not same then do feature engineering, If distinct same will do feature engineering data train
    and data test separately"""
    dist_total1, dist_total2=check_distinct(df,df2)   
    if dist_total1!=dist_total2:
        Label_df=df.select('Id', 'label')
        df_final2=df.drop('label')
        all_df =df_final2.union(df2)
        all_df_feat=feature_engineering(all_df)
        id_train=df.select('Id').collect()
        id_train=[r['Id'] for r in id_train]
        id_test=df2.select('Id').collect()
        id_test=[r['Id'] for r in id_test]
        a=all_df_feat.filter(all_df['Id'].isin(id_train))
        b=all_df_feat.filter(all_df['Id'].isin(id_test))
        a=a.join(Label_df, 'Id')
    else:
        a=feature_engineering(df)
        b=feature_engineering(df2)        
    return a,b
    
#call function feature engineering
%time data2, test2=Main_feature_engineering(df_final, test_data)

#view result of feature engineering in data train
data2.select('Id', 'features').show(5)

#view result of feature engineering in data test
test2.select('Id', 'features').show(5)


#Split df_final to train and test, train 70% and test 30%. Define seed 24 so the random data that we split will not change.
#we can define seed with any value
data_train, data_test=data2.randomSplit([0.7,0.3], 24)

#Modelling & Evaluation in Data train
#Linear Regression
#Create logistic regression model to data train
lr = LinearRegression(featuresCol='features', labelCol='label')

#fit model to data train
lr_model = lr.fit(data_train)

# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(lr_model.coefficients))
print("Intercept: %s" % str(lr_model.intercept))

#Transform model to data test
lr_prediction= lr_model.transform(data_test)

#view id, label, prediction and probability from result of modelling
lr_prediction.select("prediction","label","features").show(5)

#Linear Regression Evaluation
Calculate R squared
lr_evaluator=RegressionEvaluator(predictionCol='prediction', metricName='r2')
print("R squared (R2) on test data=%g" % lr_evaluator.evaluate(lr_prediction))

#Calculate RMSE
lr_evaluator=RegressionEvaluator(predictionCol='prediction', metricName='rmse')
print("Root Mean Squared Error (RMSE) on linear regression model=%g" % lr_evaluator.evaluate(lr_prediction))

#another way to calculate RMSE, and result is same with syntax above
test_result=lr_model.evaluate(data_test)
print("Root Mean Squared Error (RMSE) on linear regression model=%g" % test_result.rootMeanSquaredError)

#Linear Regression With Hyper-Parameter Tuning
lr_hyper = LinearRegression(featuresCol='features', labelCol='label')


#Hyper-Parameter Tuning
paramGrid_lr = ParamGridBuilder() \
    .addGrid(lr_hyper.regParam, [0.1, 0.01]) \
    .addGrid(lr_hyper.elasticNetParam, [0.8, 0.7]) \
    .build()
crossval_lr = CrossValidator(estimator=lr_hyper,
                             estimatorParamMaps=paramGrid_lr,
                             evaluator=RegressionEvaluator(),
                             numFolds=3)
#fit model to data train
lr_model_hyper= crossval_lr.fit(data_train)

#Transform model to data test
lr_prediction_hyper= lr_model_hyper.transform(data_test)

#view label, prediction and feature from result of modelling
lr_prediction_hyper.select("prediction","label","features").show(5)

#Linear Regression With Hyper-Parameter Tuning Evaluation
#Calculate RMSE
eval_rmse=RegressionEvaluator(metricName="rmse")
print("Root Mean Squared Error (RMSE) on linear regression model=%g" % eval_rmse.evaluate(lr_prediction_hyper))

#Calculate Rsquared
eval_r2=RegressionEvaluator(metricName="r2")
print("R squared (R2) on linear regression model=%g" % eval_r2.evaluate(lr_prediction_hyper))


#Decision Tree Regression
#Create Decision Tree model regression
dt = DecisionTreeRegressor(featuresCol ='features', labelCol = 'label')

#Fit model to data train
dt_model = dt.fit(data_train)

#Make prediction on data test
dt_prediction = dt_model.transform(data_test)

#View result with column selection
dt_prediction.select("prediction","label","features").show(5)

#Decision Tree Regression Evaluation
#Calculate R squared
dt_evaluator=RegressionEvaluator(predictionCol='prediction', labelCol="label", metricName='r2')
print("R squared (R2) on Decision Tree Model=%g" % dt_evaluator.evaluate(dt_prediction))

#Calculate RMSE
dt_evaluator=RegressionEvaluator(predictionCol='prediction', labelCol="label", metricName='rmse')
print("Root Mean Squared Error (RMSE) on Decision Tree Model=%g" % dt_evaluator.evaluate(dt_prediction))

#Decision Tree Regression with Hyper-Parameter Tuning
#Create Decision Tree Model
dt_hyper=DecisionTreeRegressor(featuresCol = 'features', labelCol = 'label')

#Hyper-Parameter Tuning
paramGrid_dt = ParamGridBuilder() \
    .addGrid(dt_hyper.maxDepth, [5, 7]) \
    .addGrid(dt_hyper.maxBins, [10,20]) \
    .build()
crossval_dt = CrossValidator(estimator=dt_hyper,
                             estimatorParamMaps=paramGrid_dt,
                             evaluator=RegressionEvaluator(),
                             numFolds=5)
#fit model to data train
dt_model_hyper = crossval_dt.fit(data_train)

#Transform model to data test
dt_prediction_hyper= dt_model_hyper.transform(data_test)

#View prediction, label and featues from prediction
dt_prediction_hyper.select("prediction","label","features").show(5)

#Decision Tree Regression with Hyper-Parameter Tuning Evaluation
#Calculate Rsquared
eval_r2=RegressionEvaluator(metricName="r2")
print("R squared (R2) on Decision Tree Model=%g" % eval_r2.evaluate(dt_prediction_hyper))

#Calculate RMSE
eval_rmse=RegressionEvaluator(metricName="rmse")
print("Root Mean Squared Error (RMSE) on Decision Tree Model=%g" % eval_rmse.evaluate(dt_prediction_hyper))

#Random Forest Regression
#Create Random forest model regression
rf = RandomForestRegressor(featuresCol ='features', labelCol = 'label')

#Fit model to data train
rf_model = rf.fit(data_train)

#Make prediction on data test
rf_prediction = rf_model.transform(data_test)

#View result with column selection
rf_prediction.select("prediction","label","features").show(5)

#Random Forest Regression Evaluation
#Calculate R squared
rf_evaluator=RegressionEvaluator(predictionCol='prediction', labelCol="label", metricName='r2')
print("R squared (R2) on Random Forest Model=%g" % rf_evaluator.evaluate(rf_prediction))

#Calculate RMSE
rf_evaluator=RegressionEvaluator(predictionCol='prediction', labelCol="label", metricName='rmse')
print("Root Mean Squared Error (RMSE) on Random Forest Model=%g" % rf_evaluator.evaluate(rf_prediction))

#Random Forest Regression with Hyper Parameter Tuning
#define random forest regressor
rf_hyper= RandomForestRegressor(featuresCol='features', labelCol="label")

# Hyper-Parameter Tuning
paramGrid_rf = ParamGridBuilder() \
    .addGrid(rf_hyper.numTrees, [40, 60, 80, 100]) \
    .build()
crossval_rf = CrossValidator(estimator=rf_hyper,
                             estimatorParamMaps=paramGrid_rf,
                             evaluator=RegressionEvaluator(),
                             numFolds=3) 
#fit model to data train
rf_model_hyper=crossval_rf.fit(data_train)

#Transform model to data test
rf_prediction_hyper= rf_model_hyper.transform(data_test)

#View result of prediction, label, features
rf_prediction_hyper.select("prediction","label","features").show(5)

#Random Forest Regression with Hyper Parameter Tuning Evaluation
#Calculate R squared
rf_evaluator=RegressionEvaluator(predictionCol='prediction', labelCol="label", metricName='r2')
print("R squared (R2) on Random Forest Model=%g" % rf_evaluator.evaluate(rf_prediction_hyper))

#Calculate RMSE
rf_evaluator=RegressionEvaluator(predictionCol='prediction', labelCol="label", metricName='rmse')
print("Root Mean Squared Error (RMSE) on Random Forest Model=%g" % rf_evaluator.evaluate(rf_prediction_hyper))

#Gradient Boosted Tree Regression
#Create Gradient Boosted Tree regression
gbt = GBTRegressor(featuresCol ='features', labelCol = 'label', maxIter=15)

#Fit model to data train
gbt_model = gbt.fit(data_train)

#Make prediction on data test
gbt_prediction = gbt_model.transform(data_test)

#View result of prediction, label, features
gbt_prediction.select("prediction","label","features").show(5)

#Gradient Boosted Tree Regression Evaluation
#Calculate R squared
gbt_evaluator=RegressionEvaluator(predictionCol='prediction', labelCol="label", metricName='r2')
print("R squared (R2) on Gradient Boosted Model=%g" % gbt_evaluator.evaluate(gbt_prediction))

#Calculate RMSE
gbt_evaluator=RegressionEvaluator(predictionCol='prediction', labelCol="label", metricName='rmse')
print("Root Mean Squared Error (RMSE) on Gradient Boosted Model=%g" % gbt_evaluator.evaluate(gbt_prediction))

#Gradient Boosted Tree Regression with Hyper Parameter Tuning
#Create Gradient Boosted Tree regression
gbt_hyper = GBTRegressor(featuresCol ='features', labelCol = 'label')

# Hyper-Parameter Tuning
paramGrid_gbt = ParamGridBuilder() \
    .addGrid(gbt_hyper.maxIter, [10,20])\
    .addGrid(gbt_hyper.maxDepth, [10, 12,15]) \
    .build()
crossval_gbt = CrossValidator(estimator=gbt_hyper,
                             estimatorParamMaps=paramGrid_gbt,
                             evaluator=RegressionEvaluator(),
                             numFolds=3)
#Fit Model to data train
gbt_model_hyper = crossval_gbt.fit(data_train)

#Transform model to data test
gbt_prediction_hyper= gbt_model_hyper.transform(data_test)

#View result of prediction, label, features
gbt_prediction_hyper.select("prediction","label","features").show(5)

#Gradient Boosted Tree Regression with Hyper Parameter Tuning Evaluation
#Calculate R squared
gbt_evaluator=RegressionEvaluator(predictionCol='prediction', labelCol="label", metricName='r2')
print("R squared (R2) on Gradient Boosted Model=%g" % gbt_evaluator.evaluate(gbt_prediction_hyper))

#Calculate RMSE
gbt_evaluator=RegressionEvaluator(predictionCol='prediction', labelCol="label", metricName='rmse')
print("Root Mean Squared Error (RMSE) on Gradient Boosted Model=%g" % gbt_evaluator.evaluate(gbt_prediction_hyper))


#Implementation Modelling to data test
#Prediction using Random Forest with hyper parameter tuning
#Transform model to data test
predic= rf_model_hyper.transform(test2)

#View result of prediction, label, features
predic.select("Id", "prediction","features").show(5)

#select Id and prediction column
my_submission=predic.select("Id","prediction")

#convert to Pandas dataframe
my_submission2=my_submission.toPandas()

#save to csv
my_submission2.to_csv('E:/my_submission4.csv', index = False, header = True)
