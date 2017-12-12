package gbt


import org.scalatest._


class DecisionTreeSpec extends FlatSpec with Matchers {

  final val emptyTree = Vector.fill[Double](8)(0.0)

  final val data = DataSet(
    Vector(
      Vector(0.0, 0.0),
      Vector(0.0, 0.0),
      Vector(1.0, 1.0),
      Vector(1.5, 1.0),
      Vector(1.5, 0.0),
      Vector(1.5, 0.0),
      Vector(2.0, 0.0),
      Vector(2.5, 0.0)
    ),
    Vector(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0)
  )

  final val ExpectedTree = DecisionNode(
    Decision(1, 0.5),
    DecisionNode(
      Decision(0, 1.75),
      LeafNode(-1.0),
      LeafNode(0.6666666666666666)
    ),    
    LeafNode(0.6666666666666666)
  )

  final val logit = new Logistic

  "A decision tree learner" should "construct a tree for the supplied data" in {
    val learner = new TreeConstructor(10, 1, 1.0, 0.0)
    val tree = learner.construct(
      data, data.stats(emptyTree, logit), (0 until emptyTree.size).toVector, 0
    )
    tree should equal (ExpectedTree)
  }

  "A decision tree" should "classify examples correctly" in {
    val predictions = data.x.map(v => if(logit.output(ExpectedTree(v)) > 0.5) 1.0 else 0.0)
    val correct = predictions.zip(data.y).collect{ case (p, y) if p == y => 1.0 }.sum
    correct == data.x.size
  }

}
