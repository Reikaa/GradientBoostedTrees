package gbt


case class Statistics(g: Vector[Double], h: Vector[Double]) {

  def scale(x: Double): Statistics =
    Statistics(g.map(_ * x), h.map(_ * x * x))

}


case class DataSet(x: Vector[Vector[Double]], y: Vector[Double]) {

  def groupBy(partition: Vector[Int])(condition: Vector[Double] => Boolean): (Vector[Int], Vector[Int]) = {
    val left  = partition.collect { case i if condition(x(i)) => i}
    val right = partition.collect { case i if !condition(x(i)) => i}
    (left, right)
  }

  def stats(predictions: Vector[Double], loss: Loss): Statistics =
    Statistics(      
      predictions.zip(y).map {case(z, y) => loss.gradient(y, z)},
      predictions.zip(y).map {case(z, y) => loss.hessian(y, z)}
    )
  
  def stats(predictor: Vector[Double] => Double, loss: Loss): Statistics = {
    val predictions = x.map(predictor)
    stats(predictions, loss)
  }

  def colum(c: Int): Vector[Double] = x.map(row => row(c))

  def dim: Int = x(0).size

}
