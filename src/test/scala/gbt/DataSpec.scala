package gbt


import org.scalatest._


class DataSpec extends FlatSpec with Matchers {

  final val data = DataSet(
    Vector(
      Vector(0.0, 0.0),
      Vector(0.0, 0.0),
      Vector(1.0, 0.0),
      Vector(1.0, 0.0)
    ),
    Vector(0.0, 1.0, 1.0, 0.0)
  )

  "The statistics" should "be scalable" in {
    val scaled = Statistics(Vector(1.0), Vector(1.0)).scale(0.1)
    (scaled.g(0) * 10).toInt  should be (1.0)
    (scaled.h(0) * 100).toInt should be (1.0)
  }

  "A dataset" should "be groupable" in {
    val condition = (x: Vector[Double]) => x(0) < 1.0
    val (l, r) = data.groupBy(Vector(0, 1, 2, 3))(condition)
    l should equal(Vector(0, 1))
    r should equal(Vector(2, 3))
  }

  it should "compute the statitics for a predictor" in {
    val predictor = (x: Vector[Double]) => if(x(0) < 1.0) Double.PositiveInfinity else Double.NegativeInfinity
    val stats = data.stats(predictor, new Logistic)
    stats.g should equal(Vector(1.0 , 0.0, -1.0, 0.0))
    stats.h should equal(Vector(0.0 , 0.0, 0.0, 0.0))
  }

  it should "also be indexable by colum" in {
    data.colum(0) should equal (Vector(0.0, 0.0, 1.0, 1.0))
  }

  it should "return the correct dimension" in {
    data.dim should be (2)
  }
}
