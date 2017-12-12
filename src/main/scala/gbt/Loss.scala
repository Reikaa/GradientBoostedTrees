package gbt

import math._

trait Loss {

  def gradient(truth: Double, prediction: Double): Double

  def hessian(truth: Double, prediction: Double): Double

  def loss(truth: Double, prediction: Double): Double

  def output(x: Double): Double

}


class Logistic extends Loss {

  private def sigmoid(x: Double): Double = 1.0 / (1.0 + exp(-x))

  override def loss(truth: Double, prediction: Double): Double = {
    val isOne  = -truth * log(sigmoid(prediction))
    val isZero = (1 - truth) * log(1.0 - sigmoid(prediction))
    (isOne - isZero)
  }

  override def gradient(truth: Double, prediction: Double): Double = sigmoid(prediction) - truth

  override def hessian(truth: Double, prediction: Double): Double =
    sigmoid(prediction) * (1.0 - sigmoid(prediction))

  override def output(x: Double) = sigmoid(x)

}


class Squared extends Loss {

  override def loss(truth: Double, prediction: Double): Double = pow(truth - prediction, 2.0)

  override def gradient(truth: Double, prediction: Double): Double = 2.0 * (prediction - truth)

  override def hessian(truth: Double, prediction: Double): Double = 2.0

  override def output(x: Double): Double = x

}


object LossUtils {

  def score(gradients: Vector[Double], hessians: Vector[Double], lambda: Double): Double = {
    val G = gradients.sum
    val H = hessians.sum 
    - G / (H + lambda)
  }

  def gain(
    gradients: Vector[Double],
    hessians: Vector[Double],
    left: Vector[Int],
    right: Vector[Int],
    lambda: Double,
    gamma: Double
  ): Double = {
    val GL = left.map(i => gradients(i)).sum
    val HL = left.map(i => hessians(i)).sum
    val GR = right.map(i => gradients(i)).sum
    val HR = right.map(i => hessians(i)).sum 
    0.5 * (
      (pow(GL, 2) / (HL + lambda)) + (pow(GR, 2) / (HR + lambda)) - (pow(GL + GR, 2) / (HL + HR + lambda))
    ) - gamma
  }

}
