// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

import Foundation
import TensorFlow

/// Reads a file into an array of bytes.
func readFile(_ path: String) -> [UInt8] {
    let possibleFolders  = [".", "MNIST"]
    for folder in possibleFolders {
        let parent = URL(fileURLWithPath: folder)
        let filePath = parent.appendingPathComponent(path)
        guard FileManager.default.fileExists(atPath: filePath.path) else {
            continue
        }
        let data = try! Data(contentsOf: filePath, options: [])
        return [UInt8](data)
    }
    fatalError("Filename not found: \(path)")
}

/// Reads MNIST images and labels from specified file paths.
func readMNIST(imagesFile: String, labelsFile: String) -> (images: Tensor<Float>,
                                                           labels: Tensor<Int32>) {
    print("Reading data.")
    let images = readFile(imagesFile).dropFirst(16).map(Float.init)
    let labels = readFile(labelsFile).dropFirst(8).map(Int32.init)
    let rowCount = labels.count
    let imageHeight = 28, imageWidth = 28

    print("Constructing data tensors.")
    return (
        images: Tensor(shape: [rowCount, 1, imageHeight, imageWidth], scalars: images)
            .transposed(withPermutations: [0, 2, 3, 1]) / 255, // NHWC
        labels: Tensor(labels)
    )
}

/// A classifier.
struct Classifier: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var conv1a = Conv2D<Float>(filterShape: (3, 3, 1, 32), activation: relu)
    var conv1b = Conv2D<Float>(filterShape: (3, 3, 32, 64), activation: relu)
    var pool1 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))

    var dropout1a = Dropout<Float>(probability: 0.25)
    var flatten = Flatten<Float>()
    var layer1a = Dense<Float>(inputSize: 9216, outputSize: 128, activation: relu)
    var dropout1b = Dropout<Float>(probability: 0.5)
    var layer1b = Dense<Float>(inputSize: 128, outputSize: 10, activation: softmax)

    @differentiable
    func call(_ input: Input) -> Output {
        let convolved = input.sequenced(through: conv1a, conv1b, pool1)
        return convolved.sequenced(through: dropout1a, flatten, layer1a, dropout1b, layer1b)
    }
}

let epochCount = 12
let batchSize = 100

func minibatch<Scalar>(in x: Tensor<Scalar>, at index: Int) -> Tensor<Scalar> {
    let start = index * batchSize
    return x[start..<start+batchSize]
}

let (images, numericLabels) = readMNIST(imagesFile: "train-images-idx3-ubyte",
                                        labelsFile: "train-labels-idx1-ubyte")
let labels = Tensor<Float>(oneHotAtIndices: numericLabels, depth: 10)

var classifier = Classifier()
let optimizer = RMSProp(for: classifier)

print("Beginning training...")

// The training loop.
for epoch in 1...epochCount {
    var correctGuessCount = 0
    var totalGuessCount = 0
    var totalLoss: Float = 0
    for i in 0 ..< Int(labels.shape[0]) / batchSize {
        let x = minibatch(in: images, at: i)
        let y = minibatch(in: numericLabels, at: i)
        // Compute the gradient with respect to the model.
        let 𝛁model = classifier.gradient { classifier -> Tensor<Float> in
            let ŷ = classifier(x)
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
