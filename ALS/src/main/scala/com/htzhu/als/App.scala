package com.htzhu.als

import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.sql.SparkSession
import org.jblas.DoubleMatrix

/**
  * App
  *
  * @author htzhu
  * @date 2018/11/19 15:28
  **/
object App {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
      .setAppName("ALS")
      .setMaster("local[*]")
      .set("spark.executor.memory", "10G")
      .set("spark.driver.memory", "3G")

    val spark = SparkSession.builder().config(conf).getOrCreate()
    import spark.implicits._

    // mid::title::genres
//    val movieRDD = spark.sparkContext.textFile("data/ml-10M100K/movies.dat")
//      .map(line => {
//        val data = line.split("::")
//        Movie(data(0).toInt, data(1), data(2))
//      })

    // uid::mid::rating::timestamp
    val movieRatingRDD = spark.sparkContext.textFile("data/ml-10M100K/ratings.dat")
      .map(line => {
        val data = line.split("::")
        MovieRating(data(0).toInt, data(1).toInt, data(2).toDouble, data(3).toLong)
      })

    movieRatingRDD.cache()

    // uid
    val users = movieRatingRDD.map(_.uid).distinct()
    // mid
    val movies = movieRatingRDD.map(_.mid).distinct()

    // 10, 5, 0.01 --> 0.7431148854171586
    // 12, 8, 0.01 --> 0.7213300224369459
    // 10, 10, 0.01 --> 0.7302135003371008
    // 10, 8, 0.01 --> 0.7328689882788335
    val (rank, iterations, lambda) = (10, 5, 0.01)

    // org.apache.spark.mllib.recommendation.Rating
    val ratingRDD = movieRatingRDD.map(d => Rating(d.uid, d.mid, d.rating))

    // 训练模型
    val moduel = ALS.train(ratingRDD, rank, iterations, lambda)

    val userMovie = movieRatingRDD.map(d => (d.uid, d.mid))
    val predict = moduel.predict(userMovie)

    // 将结果数据和原数据合并，计算结果数据和原始数据的方均根差
    // 方均根差越小越接近越准确，通过调整 train 参数
    val movieUserRating = movieRatingRDD.map(x => ((x.uid, x.mid), x.rating))
    val resAndSour = predict.map {
      case Rating(uid, mid, rating) => ((uid, mid), rating)
    }.join(movieUserRating)

    val rmse = math.sqrt(resAndSour.map {
      case ((uid, mid), (r1, r2)) =>
        val err = (r1 - r2)
        err * err
    }.mean())

    println("___--------" + rmse)


    // 用户矩阵
    val userRes = predict.map(d => {
      (d.user, (d.product, d.rating))
    }).groupByKey()
      .map {
        case (uid, res) =>
          (
            uid,
            res.toList.sortWith(_._2 > _._2).take(10)
            //                .map(x => x._1 + ":" + x._2).mkString("|")
          )
      }

    // 电影特征矩阵
    val movieFeatures = moduel.productFeatures.map {
      case (mid, features) => (mid, new DoubleMatrix(features))
    }

    val movieRes = movieFeatures.cartesian(movieFeatures)
      .filter { case (a, b) => a._1 != b._1 }
      .map {
        case (a, b) =>
          val simScore = this.consinSim(a._2, b._2)
          (a._1, (b._1, simScore))
      }.groupByKey()
      .map {
        case (mid, items) => (mid, items.toList)
      }

    userRes.saveAsTextFile("data/result/als/user_res")
    movieRes.saveAsTextFile("data/result/als/movie_res")

  }

  // 计算两个电影的余弦相似度
  def consinSim(movie1: DoubleMatrix, movie2: DoubleMatrix): Double = {
    movie1.dot(movie2) / (movie1.norm2() * movie2.norm2())
  }

}
