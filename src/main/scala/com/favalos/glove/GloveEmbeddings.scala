package com.favalos.glove

import scala.io.Source
import breeze.linalg._

class GloveEmbeddings(fileName: String) {

  val (wordToIdx, idxToWord, embeddings) = load(fileName)

  def findNearest(word: String, n: Int = 10): Array[String] = {

    val wordVector = wordToVector(word)

    wordVector match {

      case Some(vector) => {

        val distance = calculateDistance(vector, embeddings)
        distance.toArray
          .zipWithIndex
          .sortWith((x, y) => x._1 > y._1)
          .take(n)
          .map(i => idxToWord(i._2))
      }
    }
  }


  /* Calculate the cosine distance from one vector to all the matrix */
  def calculateDistance(denseVector: Transpose[DenseVector[Double]], denseMatrix: DenseMatrix[Double]) = {

    denseMatrix * denseVector.t
  }

  /* Word to embedding vector */
  def wordToVector(word: String): Option[Transpose[DenseVector[Double]]] = {

    wordToIdx.get(word.toLowerCase) match {

      case Some(idx) =>

        Some(embeddings(idx, ::))

      case None =>

        None
    }
  }

  /* Read vectors from file and store data in memory */
  def load(fileName: String): (Map[String, Int], Map[Int, String], DenseMatrix[Double]) = {

    val file = Source.fromFile(fileName)

    val lines = file.getLines().map { line =>
      line.split(" ")
    }.toList

    val denseMatrix = DenseMatrix(lines: _*)

    val wordToIdx = denseMatrix(::, 0).toArray.zipWithIndex.toMap
    val idxToWord = wordToIdx.toList.map(v => (v._2,  v._1)).toMap
    val embeddings = denseMatrix(::, 1 to -1).map(_.toDouble)

    (wordToIdx, idxToWord, embeddings)
  }

}

object GloveEmbeddings extends App {


  val glove = new GloveEmbeddings("/home/favalos/Downloads/glove.6B.50d.txt")


  glove.findNearest("toad").foreach(println)
  glove.findNearest("Mexico").foreach(println)
  glove.findNearest("work").foreach(println)

}
