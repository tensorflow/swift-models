#!/usr/bin/env swift -O
import Python
import TensorFlow

public func readMnist(_ filename: String) -> (Tensor<Float>, Tensor<Int32>) {
  let gzip = Python.import("gzip")
  let pickle = Python.import("cPickle")

  print("Reading data.")
  let file = gzip.open.call(with: filename, "rb")
  let serialized = pickle.load.call(with: file).tuple4
  let (pyImages, pyLabels, pyRows, pyColumns) = serialized

  print("Converting data to Swift.")
  let images: [Float] = Array(pyImages)!
  let labels: [Int32] = Array(pyLabels)!
  let rowCount = Int32(pyRows)!
  let columnCount = Int32(pyColumns)!

  print("Constructing data tensors.")
  let imagesTensor = Tensor(shape: [rowCount, columnCount], scalars: images)
  let labelsTensor = Tensor(labels)
  return (imagesTensor.toDevice(), labelsTensor.toDevice())
}

func main() {
  // Training data
  // - Note: There are two options for MNIST data for open-source.

  // - Option 1: Use unmodified MNIST data from:
  //   http://deeplearning.net/tutorial/gettingstarted.html.
  // let (images, numericLabels) = readMnistOriginal("mnist.pkl.gz")
  // let labels = Tensor<Float>(oneHotAtIndices: numericLabels, depth: 10)

  // - Option 2: Use preprocessed MNIST data.
  let (images, numericLabels) = readMnist("mnist.1000.pkl.gz")
  let labels = Tensor<Float>(oneHotAtIndices: numericLabels, depth: 10)

  // Hyper-parameters
  let iterationCount: Int32 = 10
  let learningRate: Float = 0.2
  var loss = Float.infinity

  // Parameters
  var w1 = Tensor<Float>(randomUniform: [784, 30])
  var w2 = Tensor<Float>(randomUniform: [30, 10])
  var b1 = Tensor<Float>(zeros: [1, 30])
  var b2 = Tensor<Float>(zeros: [1, 10])

  // Training loop
  var i: Int32 = 0
  repeat {
    // Forward pass
    let z1 = images ⊗ w1 + b1
    let h1 = sigmoid(z1)
    let z2 = h1 ⊗ w2 + b2
    let predictions = sigmoid(z2)

    // Backward pass
    let dz2 = predictions - labels
    let dw2 = h1.transposed(withPermutations: 1, 0) ⊗ dz2
    let db2 = dz2.sum(squeezingAxes: 0)
    let dz1 = dz2.dot(w2.transposed(withPermutations: 1, 0)) * h1 * (1 - h1)
    let dw1 = images.transposed(withPermutations: 1, 0) ⊗ dz1
    let db1 = dz1.sum(squeezingAxes: 0)

    // Gradient descent
    w1 -= dw1 * learningRate
    b1 -= db1 * learningRate
    w2 -= dw2 * learningRate
    b2 -= db2 * learningRate

    // Update current loss
    loss = dz2.squared().mean(squeezingAxes: 1, 0).scalarized()

    // Update iteration count
    i += 1
  } while i < iterationCount

  // Check results
  print("Loss: \(loss)")
}

main()
