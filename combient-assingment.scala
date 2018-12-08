// Databricks notebook source
val session = org.apache.spark.sql.SparkSession.builder
        .appName("Combient app")
        .getOrCreate; //create Spark session

val trainDf = session.read
        .format("com.databricks.spark.csv")
        .option("header", "true") //file has headers else headers taken as first data row
        .load("/FileStore/tables/ftrl22kd1487181764148/train.csv"); //read in training data from dbfs

// COMMAND ----------

/* Data transformation */

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.linalg.Vectors

val labelIndexer = new StringIndexer()
  .setInputCol("TARGET")
  .setOutputCol("label")
  .fit(trainDf)

val d2 = labelIndexer.transform(trainDf).drop("ID").drop("TARGET") //convert string TARGET column to double lable column for predicting and drop not needed columns
  
val dataRdd = d2.map(r =>
  Tuple2(
    r.getAs[Double]("label"),
    Vectors.dense(r.toSeq.dropRight(1).map(x => x.toString.toDouble).toArray)
  ))

val data = dataRdd.rdd //sample data due to high bias
      .sampleByKey(withReplacement = true, Map(1.0->2.0, 0.0->0.1), seed = 1024L)
      .toDF("label", "features")


// COMMAND ----------

import org.apache.spark.ml.classification.RandomForestClassifier

val rf = new RandomForestClassifier()
  .setLabelCol("label")
  .setFeaturesCol("features")
  .setNumTrees(15)

val model = rf.fit(data) //fit model

// COMMAND ----------

import org.apache.spark.sql.types._

val testDf = session.read
        .format("com.databricks.spark.csv")
        .option("header", "true") //file has headers else headers taken as first data row
        .load("/FileStore/tables/cqr8c8pu1487409991112/test.csv"); //read in test data from dbfs

val testData = testDf.map(r =>
  Tuple2(
    r.getAs[String]("ID"),
    Vectors.dense(r.toSeq.drop(1).map(x => x.toString.toDouble).toArray)
  )).toDF("ID", "features")

val predictions = model.transform(testData)

predictions.select(predictions("ID"), predictions("prediction").cast(IntegerType).as("TARGET")).write
  .format("com.databricks.spark.csv")
  .option("header", "false")
  .save("/FileStore/combient/2to0.1sampling/results")

predictions.show()

// COMMAND ----------

/*Testing model on training data because in training data results are known */
/*import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.classification.RandomForestClassifier

val dataTest = dataRdd.rdd
      .sampleByKey(withReplacement = true, Map(1.0->2.0, 0.0->0.1), seed = 1024L)
      .toDF("label", "features")

val splits = dataTest.randomSplit(Array(0.8, 0.2))
val (trainingData, testData) = (splits(0), splits(1))

val rfTest = new RandomForestClassifier()
  .setLabelCol("label")
  .setFeaturesCol("features")
  .setNumTrees(15)

val testModel = rfTest.fit(trainingData) //fit model

val predictions = testModel.transform(testData)

val predictionAndLabels = predictions.select("prediction", "label").rdd.map {
   case Row(prediction: Double, label: Double) =>
     (prediction, label)
}

val metrics = new MulticlassMetrics(predictionAndLabels)

// Confusion matrix
println("Confusion matrix:")
println(metrics.confusionMatrix)

println("F1 measure:")
println(metrics.weightedFMeasure)

println("Accuracy:")
println(metrics.accuracy)*/
