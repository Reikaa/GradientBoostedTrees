package gbt


import org.scalatest._


class DecisionSpec extends FlatSpec with Matchers {

  "The midpoint generator" should "compute all mid points" in {
    Decision.midPoints(Vector(1.0, 1.0, 2.0, 3.0, 4.0)) should equal (Vector(1.5, 2.5, 3.5))
  }

  "The hypothesis generator" should "compute all hypothesis" in {
    val data = DataSet(
      Vector(
        Vector(1.0, 5.0),
        Vector(2.0, 6.0),
        Vector(3.0, 7.0),
        Vector(4.0, 8.0)
      ),
      Vector(0.0, 1.0, 1.0, 0.0)
    )
    Decision.generate(data, Vector(0,1,2)) should equal (Set(
      Decision(0, 1.5), Decision(0, 2.5),
      Decision(1, 5.5), Decision(1, 6.5))
    )
  }

}
