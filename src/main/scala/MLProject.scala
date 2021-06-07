package si

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession

object MLProject {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setMaster("local[8]")
      .setAppName("MLProject")
      .set("spark.driver.host", "127.0.0.1")

    val spark = SparkSession.builder().config(conf).getOrCreate()

    // load training data (60k images 28x28)
    val trainingData = spark.read.format("libsvm")
      .option("numFeatures", "784")
      .load("mnist_train.txt")

    // load test data (10k images 28x28)
    val testData = spark.read.format("libsvm")
      .option("numFeatures", "784")
      .load("mnist_test.txt")

    val nn = new MultilayerPerceptronClassifier()

    // create grids of parameters that will be checked
    // to find the most optimal way of training the MLP
    val paramGrid = new ParamGridBuilder()
      .addGrid(nn.maxIter, Array(50, 70))
      .addGrid(nn.layers, Array(
        Array(784, 500, 300, 100, 10),
        Array(784, 300, 100, 100, 10))
      )
      .addGrid(nn.stepSize, Array(0.03, 0.05))
      .build()

    val cv = new CrossValidator()
      .setEstimator(nn)
      .setEvaluator(new MulticlassClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setParallelism(8)

    // find the most optimal params for the MLP
    val trainedModel = cv.fit(trainingData)

    // extract and print the best params
    val paramMaps = trainedModel.bestModel.extractParamMap()
    val bestLayers: Array[Int] = paramMaps
      .apply(trainedModel.bestModel.getParam("layers")).asInstanceOf[Array[Int]]
    val bestMaxIter = paramMaps
      .apply(trainedModel.bestModel.getParam("maxIter"))
    val bestStepSize = paramMaps
      .apply(trainedModel.bestModel.getParam("stepSize"))
    println("Best Layers: " + bestLayers.mkString("Array(", ", ", ")"))
    println("Best MaxIter: " + bestMaxIter)
    println("Best StepSize: " + bestStepSize)
    println("------")

    // print scores of all of the param combinations
    for (metric <- trainedModel.avgMetrics) {
      println(metric)
    }

    // show first 20 predictions of our model (on test data)
    trainedModel.transform(testData).show(20)

    spark.stop()
  }
}
