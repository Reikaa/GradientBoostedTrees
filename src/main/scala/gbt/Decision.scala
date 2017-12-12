package gbt


case class Decision(col: Int, th: Double) {

  override def toString(): String = s"x[$col] <= $th"

  def condition(x: Vector[Double]): Boolean = x(col) < th

}


object Decision {

  def midPoints(x: Vector[Double]): Vector[Double] = {
    x.sorted.sliding(2).collect { case x if x.size == 2 && x(1) - x(0) > 0.0 =>
      ((x(1) - x(0)) / 2) + x(0)
    }.toVector
  }

  def generate(data: DataSet, partition: Vector[Int]): Set[Decision] = {
    val h = for(attr <- 0 until data.dim) yield {
      val col = data.colum(attr)
      val x = partition.map(i => col(i))
      midPoints(x).map(th => new Decision(attr, th))
    }
    h.flatten.toSet
  }

}

