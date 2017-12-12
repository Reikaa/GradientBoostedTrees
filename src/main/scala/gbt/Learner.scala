package gbt

import scala.io.Source
import util.Random
import java.io.FileWriter

class Learner(
  depth: Int, iter: Int, rate: Double, bias: Double, minSup: Int,
  penaltyTree: Double, penaltyL2: Double, loss: Loss
) {

  import Learner._

  private def learn(trainSet: DataSet): LearningProblem = {
    val learner = new TreeConstructor(depth, minSup, penaltyL2, penaltyTree)
    val booster = new GradientBoosting(learner)
    val problem = booster.init(trainSet, iter, rate, bias, loss)
    val learned = booster.learn(problem)
    learned
  }
  
  def run(file: String): (Double, Double, LearningProblem) = {
    val data = Source.fromFile(file).getLines.collect {
      case line if !isHeader(line) => parse(line)
    }.toVector
    val isTrain = (for(i <- 0 until data.size) yield rand.nextDouble < 0.66).toVector
    val train = data.zip(isTrain).collect {case x if x._2  => x._1 }
    val test = data.zip(isTrain).collect {case x if !x._2 => x._1 }
    val trainSet = DataSet(train.map(_._1), train.map(_._2))
    val testSet = DataSet(test.map(_._1), test.map(_._2))
    val learned = learn(trainSet)
    val accTest = eval(testSet, learned)
    val accTrain = eval(trainSet, learned)
    (accTest, accTrain, learned)
  }


  def eval(set: DataSet, learned: LearningProblem): Double = {
    val predictions = set.x.map(x => loss.output(learned.predict(x)))
    if(loss.isInstanceOf[Logistic]) {
      val eval = predictions.map(x =>
        if(x > 0.5) 1.0 else 0.0
      ).zip(set.y)
      val correct = eval.map {case (p, y) => if (p == y) 1.0 else 0.0}.sum
      correct / set.y.size
    } else {
      val error = predictions.zip(set.y).map { case (p,y) => Math.pow(p - y, 2.0)}.sum
      Math.sqrt(error / predictions.size)
    }
  }

}

object Learner {

  final val rand = new Random(0)

  def apply(
    depth: Int, iter: Int, rate: Double, bias: Double, minSup: Int,
    penaltyTree: Double, penaltyL2: Double, loss: Loss
  ): Learner = new Learner(
    depth, iter, rate, bias, minSup,
    penaltyTree, penaltyL2, loss
  )

  def isHeader(line: String): Boolean = line.contains("V")

  def parse(line: String): (Vector[Double], Double) = {
    val cmp = line.split(",")    
    val label = if (cmp.last.contains(".")) cmp.last.toDouble else cmp.last.toInt.toDouble
    val vec = cmp.slice(0, cmp.length - 1).map(_.toDouble).toVector
    (vec, label)
  }

}
