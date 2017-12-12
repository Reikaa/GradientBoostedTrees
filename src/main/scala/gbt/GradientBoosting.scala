package gbt


case class LearningProblem(
  data: DataSet,
  predictions: Vector[Double],
  losses: List[Double],
  ensemble: List[Node],
  iterLeft: Int,
  learningRate: Double,
  bias: Double,
  loss: Loss
) {

  def partition(): Vector[Int] = (0 until data.x.size).toVector

  def stats(): Statistics = data.stats(predictions, loss).scale(learningRate)

  private def predict(
    x: Vector[Double], e: List[Node]
  ): Double = e.map(t => t(x)).sum + bias

  def predict(x: Vector[Double]): Double =
    predict(x, ensemble)

  def update(tree: Node): LearningProblem = {
    val newEnsemble = tree :: ensemble
    val newPredictions = data.x.map(x => predict(x, newEnsemble))
    val newLosses = data.y.zip(newPredictions).map {
      case (y, p) => loss.loss(y,p)
    }.sum
    LearningProblem(
      data,
      newPredictions,
      newLosses :: losses,
      newEnsemble,
      iterLeft - 1,
      learningRate,
      bias,
      loss
    )
  }

}


class GradientBoosting(learner: TreeConstructor) {

  def init(data: DataSet, nIter: Int, rate: Double, bias: Double, loss: Loss): LearningProblem = 
    LearningProblem(
      data,
      Vector.fill[Double](data.x.size)(bias),
      List.empty[Double],
      List.empty[Node],
      nIter,
      rate,
      bias,
      loss
    )

  def learn(p: LearningProblem): LearningProblem = {
    val tree = learner.construct(p.data, p.stats, p.partition, 0)
    if (p.iterLeft == 0) p.update(tree)
    else learn(p.update(tree))
  }

}

