from pyspark.sql import SQLContext
from pyspark import SparkContext, SparkConf
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import pandas as pd
import numpy as np
from dateutil.parser import parse
from datetime import datetime
from matplotlib import pyplot as plt
import time


# path = "https://asosrecssignals.blob.core.windows.net/signals/signals/receipts"
# path = "wasb://signals@asosrecssignals.blob.core.windows.net/signals/receipts"  # recursively reads all of the receipts

# define paths to data
today = datetime.now()
startday = today
productPath = "wasb://product@asosrecsoutput.blob.core.windows.net/" + today.minusDays(3).toString("yyyy/MM/dd") + "/*"
demographicspath = "wasb://customers@asosrecsoutput.blob.core.windows.net/"  + today.minusDays(3).toString("yyyy/MM/dd") + "/*"

receiptspath = "wasb://signals@asosrecssignals.blob.core.windows.net/signals/receipts/yy=2015/mm=" + today.month + "/"
productviewspathCom = "wasb://signals@asosrecssignals.blob.core.windows.net/productviews/adobe/productviews/yy=2015/mm=" + today.month + "/"
productviewspathApps = "wasb://signals@asosrecssignals.blob.core.windows.net/productviews/adobeapps/productviews/yy=2015/mm=" + today.month + "/"
productAugmentedPath = "wasb://product@asosrecstemp.blob.core.windows.net/productaugmented/2015/" + today.month + "/0" + startday

productSegmentsPath    = "wasb://productsegments@asosrecsoutput.blob.core.windows.net/2015/07/01/"
customerSegmentsPath   = "wasb://customersegments@asosrecsoutput.blob.core.windows.net/2015/07/01/"
productPolarizingPath  = "wasb://productspolarizing@asosrecsoutput.blob.core.windows.net/2015/07/01/"
productsimilarityPath  = "wasb://productsimilarity@asosrecsoutput.blob.core.windows.net/2015/07/01/"
productPath            = "wasb://product@asosrecsoutput.blob.core.windows.net/2015/07/01/"
customerPolarizingPath = "wasb://customerpolarizing@asosrecsoutput.blob.core.windows.net/2015/07/01/"
sourcelookupPath       = "wasb://oneoff@asosrecstesting.blob.core.windows.net/oneoffnew/source”


def read_responsys(pword, username='BenChamberlain', table='ResponsysClickData_View'):
    """
    Read a Responsys table from Azure SQL into a data frame
    The shell must have been started with option --jars [path_to_sqljdbc4.jar]
    :param pword: Azure SQL password - on slip in desk draw
    :param username: Azure SQL username - on slip in desk draw
    :param table: The table to read. See https://eun-su1.azuredatacatalog.com/#/browse?searchTerms=responsys for tables
    :return: a Spark DataFrame object
    """

    jdbcurl = "jdbc:sqlserver://asos-an-mkt-dw-generic-live-eun.database.windows.net;user=" + \
              username + "@asos-an-mkt-dw-generic-live-eun;password=" + \
              pword + ";database=Responsys"
    driver = "com.microsoft.sqlserver.jdbc.SQLServerDriver"
    jdbcDF = sqlContext.read.format("jdbc").options(url=jdbcurl, driver=driver, dbtable=table).load()
    return jdbcDF


def parse_created_on(row):
    """
    convert date to a time delta
    :param row: a row of the customer table
    :return: the time delta in days
    """
    row_date = parse(row)
    row_date = row_date.replace(tzinfo=None)
    diff = datetime.now() - row_date
    return diff.days


def read_customer(path):
    """
    Read customer data into SparkDF
    :param path:
    :return:
    """
    lines = sc.textFile(path, use_unicode=False)
    customers = lines.map(lambda line: line.split('\t'))
    print customers.takeOrdered(1)
    schema_list = [("id", IntegerType), ("guid", IntegerType), ("gender", StringType), ("country", StringType),
                   ("created_on", DateType), ("modified_on", DateType), ("yob", IntegerType), ("premier", IntegerType)]
    schema_str = "cust_id guid gender country created_on modified_on yob premier"
    # fields = [StructField(field_name, col_type(), True) for field_name, col_type in schema_list]
    fields = [StructField(field_name, StringType(), True) for field_name in schema_str.split()]
    schema = StructType(fields)

    udfstring_to_int = udf(lambda x: int(x), IntegerType())
    udf_tenure = udf(lambda x: parse_created_on(x), IntegerType())

    customers_df = sqlContext.createDataFrame(customers, schema)

    customers_df = customers_df.withColumn("yob", udfstring_to_int("yob"))
    customers_df = customers_df.withColumn("tenure", udf_tenure("created_on"))

    # customers_df = customers_df.withColumn("premier", customers_df.select(customers_df["premier"].cast(IntegerType()).alias("premier")))
    print customers_df.dtypes
    print customers_df.columns
    print customers_df.describe().collect()
    counts = (customers_df
              .groupBy('country')
              .count()
              .sort('count', ascending=False))

    for count in counts.take(10):
        print count

    # counts.sort('country', ascending=False)
    #     .take(10))
    #
    # for count in counts:
    #     print count
    print customers_df.freqItems(cols=["country"])
    customers_df.show()
    customers_df.printSchema()
    return customers_df


def write_to_blob(path, df):
    """
    Write out a data frame to blob storage
    :param path: eg. 'wasb://responsys@asoscustprofdevelopment.blob.core.windows.net/responsys/ClickData_View.csv'
    THE THING BEFORE THE @ IS THE HIGHEST LEVEL DIRECTORY. THE THING AFTER THE @ IS THE STORAGE ACCOUNT (CF. C://)
    :param df: a pyspark data frame
    :return:
    """
    # This doesn't work for the Responsys data, but might work for other data sets
    df.map(lambda x: "\t".join(str(elem) for elem in x)).repartition(512).saveAsTextFile(path)
    # this writes blobs, but the format is "Row1(Col1 val1 Col2 val2....) Row2(Col1 ...)"
    df.rdd.repartition(512).saveAsTextFile(path)
    df.rdd.map(lambda x: "\t".join(map(str, x))).coalesce(1).saveAsTextFile(path)


def write_local_databricks(path, df):
    """
    Write out data locally - MUST HAVE INSTALLED THE DATABRICKS PACKAGE WHEN STARTING PYSPARK
    --packages com.databricks:spark-csv_2.10:1.0.3
    :param path: eg. wasb://testing@asosrecstesting.blob.core.windows.net/variousstuff/recsbatch/signals_spark
    :param df: a pyspark data frame
    :return:
    """
    df.write.format('com.databricks.spark.csv').save(path)



def simple_cust_read(path):
    """
    test function to replicate in scala
    :param path:
    :return:
    """


def read_receipts(path, ids=None):
    """
    read receipts data from signals
    :param path:
    :return:
    """
    lines = sc.textFile(path)
    receipts = lines.map(lambda x: x.split('|'))
    print receipts.take(10)
    # print receipts.take(10)
    schema_str = """cust_id product_id variant_id division_id source_id qty date origin price discount_type used_for_recs
    date_modified"""
    fields = [StructField(field_name, StringType(), True) for field_name in schema_str.split()]
    schema = StructType(fields)
    receipts_df = sqlContext.createDataFrame(receipts, schema)
    udfstring_to_date = udf(parse_created_on, IntegerType())
    # udfstring_to_date = udf(parse, DateType())
    receipts_df = (receipts_df
                   .withColumn("delta_date", udfstring_to_date("date")))
    receipts_out = (receipts_df
                    .filter(receipts_df.qty > 1)
                    .select(receipts_df.cust_id, receipts_df.qty, receipts_df.price, receipts_df.delta_date)
                    # .pivot() not available until Spark 1.6
                    .groupby("cust_id").agg({"price": "mean", "qty": "mean", "delta_date": "mean"})
                    )
    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes.set_title('receipts')
    receipts_out.toPandas().boxplot(ax=axes)
    plt.show()
    receipts_out.show(n=30)
    return receipts_df


if __name__ == '__main__':
    # create local context with 4 way distribution
    # Create new config - The default max result size is 1GB, which is annoying!
    conf = SparkConf().set("spark.driver.maxResultSize", "18g")
    # Create new context
    sc = SparkContext(conf=conf)
    # sc = SparkContext('local[4]')
    sqlContext = SQLContext(sc)
    # set logging level
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
    # This will read all of the 2016 receipts
    receipts_path = "wasb://signals@asosrecssignals.blob.core.windows.net/signals/receipts/yy=2016/mm*/dd*/"  # recursively reads all of the receipts
    cust_path = "wasb://customers@asosrecsoutput.blob.core.windows.net/2016/04/28/part*"
    # receipts_df = read_receipts(path=receipts_path)

    customer_df = read_customer(path=cust_path)
    customer_df.toPandas().to_csv('customers.csv')
    #
    # good_cust = receipts_df.join(customer_df, on="cust_id", how="inner")
    # good_cust.show()

    # customers.columns = cust_columns
    # customers.set_index('id', inplace=True)
    # print customers.shape
    # customers_sp = sc.createDataFrame(customers)
    # customers_sp.count()
