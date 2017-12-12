package gbt


import org.scalatest._


class GradientBoostingSpec extends FlatSpec with Matchers {

  final val Data = DataSet(
    Vector(
      Vector(0.0167, 0.018, 0.0084, 0.009, 0.0032),
      Vector(0.0191, 0.014, 0.0049, 0.0052, 0.0044),
      Vector(0.0244, 0.0316, 0.0164, 0.0095, 0.0078),
      Vector(0.0073, 0.005, 0.0044, 0.004, 0.0117),
      Vector(0.0015, 0.0072, 0.0048, 0.0107, 0.0094),
      Vector(0.0089, 0.0057, 0.0027, 0.0051, 0.0062),
      Vector(0.0138, 0.0092, 0.0143, 0.0036, 0.0103),
      Vector(0.0097, 0.0085, 0.0047, 0.0048, 0.0053),
      Vector(0.0049, 0.0065, 0.0093, 0.0059, 0.0022),
      Vector(0.0013, 0.0106, 0.0127, 0.0178, 0.0231),
      Vector(0.009, 0.0242, 0.0224, 0.019, 0.0096),
      Vector(0.0187, 0.023, 0.0057, 0.0113, 0.0131),
      Vector(0.0193, 0.0032, 0.0377, 0.0126, 0.0156),
      Vector(0.0122, 0.0038, 0.0101, 0.0228, 0.0124),
      Vector(0.0074, 0.0035, 0.01, 0.0048, 0.0019),
      Vector(0.0152, 0.0052, 0.0121, 0.0124, 0.0055),
      Vector(0.0277, 0.0097, 0.0054, 0.0148, 0.0092),
      Vector(0.0129, 0.0066, 0.0044, 0.0134, 0.0092)
    ),
    Vector(
      0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
      1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0
    )
  )

  final val Learner = new TreeConstructor(5, 1, 1.0, 0.0)
  final val Gbt = new GradientBoosting(Learner)
  final val logit = new Logistic

  "Gradient boosting" should "initialize the learning problem" in {
    val problem = Gbt.init(Data, 2, 0.01, 0.0, logit)
    problem.predictions.size should be (Data.y.size)
    problem.predictions.sum should be (0.0)
  }

  it should "learn an ensemble of given size" in {
    val problem = Gbt.init(Data, 9, 1.0, 0.0, logit)
    val learned = Gbt.learn(problem)
    val eval = learned.predictions.map(x =>
      if(logit.output(x) > 0.5) 1.0 else 0.0
    ).zip(Data.y)
    val correct = eval.map {case (p, y) => if (p == y) 1.0 else 0.0}.sum
    correct.toInt should be (Data.y.size)
    learned.losses.head should be < learned.losses.last
    learned.ensemble.size should be (10)   
  }

}

