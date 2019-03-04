// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import TensorFlow
import Python

let np = Python.import("numpy")

/// Reads a file into an array of bytes.
func readFile(_ filename: String) -> [UInt8] {
    let d = Python.open(filename, "rb").read()
    return Array(numpy: np.frombuffer(d, dtype: np.uint8))!
}

/// Reads MNIST images and labels from specified file paths.
func readMNIST(imagesFile: String, labelsFile: String) -> (images: Tensor<Float>,
                                                           labels: Tensor<Int32>) {
    print("Reading data.")
    let images = readFile(imagesFile).dropFirst(16).map { Float($0) }
    let labels = readFile(labelsFile).dropFirst(8).map { Int32($0) }
    let rowCount = Int32(labels.count)
    let columnCount = Int32(images.count) / rowCount

    print("Constructing data tensors.")
    return (
        images: Tensor(shape: [rowCount, columnCount], scalars: images) / 255,
        labels: Tensor(labels)
    )
}

/// A classifier.
struct Classifier: Layer {
    var layer1 = Dense<Float>(inputSize: 784, outputSize: 30, activation: relu)
    var layer2 = Dense<Float>(inputSize: 30, outputSize: 10, activation: relu)

    @differentiable
    func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
        return layer2.applied(to: layer1.applied(to: input, in: context), in: context)
    }
}

let epochCount = 20
let batchSize = 20

func minibatch<Scalar>(in x: Tensor<Scalar>, at index: Int) -> Tensor<Scalar> {
    let start = Int32(index * batchSize)
    return x[start..<start+Int32(batchSize)]
}

let (images, numericLabels) = readMNIST(imagesFile: "train-images-idx3-ubyte",
                                        labelsFile: "train-labels-idx1-ubyte")
let labels = Tensor<Float>(oneHotAtIndices: numericLabels, depth: 10)

var classifier = Classifier()
let context = Context(learningPhase: .training)
let optimizer = SGD<Classifier, Float>(learningRate: 0.2)

// The training loop.
for epoch in 0..<epochCount {
    var correctGuessCount = 0
    var totalGuessCount = 0
    var totalLoss: Float = 0
    for i in 0 ..< Int(labels.shape[0]) / batchSize {
        let x = minibatch(in: images, at: i)
        let y = minibatch(in: numericLabels, at: i)
        // Compute the gradient with respect to the model.
        let 𝛁model = classifier.gradient { classifier -> Tensor<Float> in
            let ŷ = classifier.applied(to: x, in: context)
            let correctPredictions = ŷ.argmax(squeezingAxis: 1) .== y
            correctGuessCount += Int(Tensor<Int32>(correctPredictions).sum().scalarized())
            totalGuessCount += batchSize
            let loss = softmaxCrossEntropy(logits: ŷ, labels: y)
            totalLoss += loss.scalarized()
            return loss
        }
        // Update the model's differentiable variables along the gradient vector.
        optimizer.update(&classifier.allDifferentiableVariables, along: 𝛁model)
    }

    let accuracy = Float(correctGuessCount) / Float(totalGuessCount)
    print("""
          [Epoch \(epoch)] \
          Loss: \(totalLoss), \
          Accuracy: \(correctGuessCount)/\(totalGuessCount) (\(accuracy))
          """)
}
