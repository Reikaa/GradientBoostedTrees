package gbt


trait Node {

  def printTree(offset: String): String

  def apply(x: Vector[Double]): Double

}


case class DecisionNode(decision: Decision, left: Node, right: Node) extends Node {

  override def printTree(offset: String): String = {
    s"$offset $decision\n" +  left.printTree(offset + "    ") + right.printTree(offset + "    ")
  }

  override def apply(x: Vector[Double]): Double = {
    if(decision.condition(x)) left(x) else right(x)
  }

}


case class LeafNode(score: Double) extends Node {

  override def printTree(offset: String): String = {
    s"$offset return $score\n"
  }

  override def apply(x: Vector[Double]): Double = {
    score
  }

}


class TreeConstructor(maxDepth: Int, minSupport: Int, lambda: Double, gamma: Double) {

  def stop(gain: Double, depth: Int, support: Int): Boolean = {
    depth > maxDepth || support <= minSupport || gain < 0.0
  }

  def mkLeaf(stats: Statistics, partition: Vector[Int]): LeafNode = LeafNode(
    LossUtils.score(
      partition.map(i => stats.g(i)),
      partition.map(i => stats.h(i)),
      lambda
    )
  )

  def construct(data: DataSet, stats: Statistics, partition: Vector[Int], depth: Int): Node = {
    val hypothesis = Decision.generate(data, partition)
    if (hypothesis.isEmpty) mkLeaf(stats, partition)
    else {
      val (g, l, r, h) = hypothesis.map(h => {
        val (left, right) = data.groupBy(partition)(h.condition)
        val gain = LossUtils.gain(
          stats.g, stats.h, left, right, lambda, gamma
        )
        (gain, left, right, h)
      }).maxBy(_._1)
      if(stop(g, depth, partition.size)) mkLeaf(stats, partition)
      else DecisionNode(
        h,
        construct(data, stats, l, depth + 1),
        construct(data, stats, r, depth + 1)
      )
    }
  }

}
