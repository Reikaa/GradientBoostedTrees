package gbt

import org.scalatest._

class LearnerSpec extends FlatSpec with Matchers {

  final val logit = new Logistic
  final val squared = new Squared

  "The learner" should "classify the reduced Iris dataset 100%" in {
    val learner = Learner(3, 20, 1.0, 0.0, 1, 0.0, 1.0, logit)
    val (test, train, learned) = learner.run("data/iris.csv")
    val loss = for(i <- 1 until learned.losses.size) yield
      learned.losses(i) > learned.losses(i - 1)
    val prob = learned.predictions.map(x => logit.output(x))
    val bound = prob.map(x => x < 0.1 || x > 0.9)
    test should be (1.0)
    train should be (1.0)
    loss.reduce(_ && _) should be (true)
    bound.reduce(_ && _) should be (true)
  }

  
  "The learner" should "classify the reduced Sonar dataset 80%" in {
    val learner = Learner(10, 20, 0.1, 0.0, 1, 0.0, 1.0, logit)
    val (test, train, learned) = learner.run("data/Sonar.csv")
    val loss = for(i <- 1 until learned.losses.size) yield
      learned.losses(i) > learned.losses(i - 1)
    val prob = learned.predictions.map(x => logit.output(x))
    val bound = prob.map(x => x < 0.1 || x > 0.9)
    test should be > 0.8
    train should be (1.0)
    loss.reduce(_ && _) should be (true)
    bound.reduce(_ && _) should be (true)
  }
  

  "The learner" should "regress on a sin function" in {
    val learner = Learner(10, 20, 0.01, 0.0, 1, 0.0, 1.0, squared)
    val (test, train, learned) = learner.run("data/sin.csv")
    val loss = for(i <- 1 until learned.losses.size) yield
      learned.losses(i) > learned.losses(i - 1)
    loss.reduce(_ && _) should be (true)
    train should be < 0.06
    test should be  < 0.09
  }

}
