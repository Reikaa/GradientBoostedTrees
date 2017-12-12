package gbt


import org.scalatest._


class LossSpec extends FlatSpec with Matchers {

  import LossUtils._  
  final val logit = new Logistic
  final val squared = new Squared

  "The sigmoid loss function" should "compute the sigmoid" in {
    logit.output(0.0) should be (0.5)
    logit.output(Double.PositiveInfinity) should be (1.0)
    logit.output(Double.NegativeInfinity) should be (0.0)
  }

  it should "rate accurate solutions higher than inaccurate ones" in {
    logit.loss(1.0, 10.0) should be < logit.loss(1.0, -10.0)
    logit.loss(0.0, 10.0) should be > logit.loss(0.0, -10.0)
  }

  it should "compute the gradient" in {
    logit.gradient(0.0, 0.0) should be (0.5)
    logit.gradient(0.0, Double.PositiveInfinity) should be (1.0)
    logit.gradient(0.0, Double.NegativeInfinity) should be (0.0)
    logit.gradient(1.0, 0.0) should be (-0.5)
    logit.gradient(1.0, Double.PositiveInfinity) should be (0.0)
    logit.gradient(1.0, Double.NegativeInfinity) should be (-1.0)
  }

  it should "compute the hessian" in {
    logit.hessian(0.0, 0.0) should be (0.25)
    logit.hessian(0.0, Double.PositiveInfinity) should be (0.0)
    logit.hessian(0.0, Double.NegativeInfinity) should be (0.0)
    logit.hessian(1.0, Double.PositiveInfinity) should be (0.0)
    logit.hessian(1.0, Double.NegativeInfinity) should be (0.0)
  }

  "The squared error loss" should "compute the gradient" in {
    squared.gradient(0.0, -1.0) should be (-2.0)
    squared.gradient(0.0,  1.0) should be (2.0)
    squared.gradient(0.0,  0.0) should be (0.0)
  }

  it should "compute the hessian" in {
    squared.hessian(0.0, -1.0) should be (2.0)
    squared.hessian(0.0,  1.0) should be (2.0)
    squared.hessian(0.0,  0.0) should be (2.0)
  }

  it should "rate accurate solutions higher than inaccurate ones" in {
    squared.loss(0.0,  2.0) should be > squared.loss(0.0, 1.0)
    squared.loss(0.0, -2.0) should be > squared.loss(0.0, 1.0)
  }

  "The Loss utils" should "compute the child weight" in {
    score(Vector(1, 1),   Vector(0, 0), 1.0) should be (-2)
    score(Vector(1,-1),   Vector(0, 0), 1.0) should be (0.0)
    score(Vector(-1, -1), Vector(0, 0), 1.0) should be (2)
  }

  it should "compute the gain of two gradient/hessian sets" in {
    val g = Vector(-1.0,-1.0,-1.0, 1.0, 1.0, 1.0) 
    val h = Vector( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    val l1 = Vector(0, 1, 2)
    val r1 = Vector(3, 4, 5) 
    gain(g, h, l1, r1, 1.0, 0.0) should be (18.0 / 2.0)

    val l2 = Vector(0, 3)
    val r2 = Vector(1, 2) 
    gain(g, h, l2, r2, 1.0, 0.0) should be (0.0)
  }

}
