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

import Foundation
import TensorFlow

/// Reads MNIST images and labels from specified file paths.
public func readMnist(imagesFile: String, labelsFile: String) -> (images: Tensor<Float>, labels: Tensor<Int32>) {
    print("Reading data.")
    let imageData = try! Data(contentsOf: URL(fileURLWithPath: imagesFile)).dropFirst(16)
    let labelData = try! Data(contentsOf: URL(fileURLWithPath: labelsFile)).dropFirst(8)
    let images = imageData.map { Float($0) }
    let labels = labelData.map { Int32($0) }
    let rowCount = Int32(labels.count)
    let columnCount = Int32(images.count) / rowCount

    print("Constructing data tensors.")
    let imagesTensor = Tensor(shape: [rowCount, columnCount], scalars: images)
    let labelsTensor = Tensor(labels)
    return (imagesTensor.toAccelerator(), labelsTensor.toAccelerator())
}

/// Parameters of an MNIST classifier.
struct MNISTParameters : ParameterAggregate {
  var w1 = Tensor<Float>(randomUniform: [784, 30])
  var w2 = Tensor<Float>(randomUniform: [30, 10])
  var b1 = Tensor<Float>(zeros: [1, 30])
  var b2 = Tensor<Float>(zeros: [1, 10])
}

/// Train a MNIST classifier for the specified number of iterations.
func train(_ parameters: inout MNISTParameters, iterationCount: Int) {
    // Get script directory. This is necessary for MNIST.swift to work when
    // invoked from any directory.
    let currentDirectory =
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let currentScriptPath = URL(fileURLWithPath: CommandLine.arguments[0],
                                relativeTo: currentDirectory)
    let scriptDirectory = currentScriptPath.appendingPathComponent("..")

    // Get training data.
    let imagesFile =
        scriptDirectory.appendingPathComponent("train-images-idx3-ubyte").path
    let labelsFile =
        scriptDirectory.appendingPathComponent("train-labels-idx1-ubyte").path
    let (images, numericLabels) = readMnist(imagesFile: imagesFile,
                                            labelsFile: labelsFile)
    let labels = Tensor<Float>(oneHotAtIndices: numericLabels, depth: 10)
    // FIXME: Defining batchSize as a scalar, or as a tensor as follows instead
    // of returning it from readMnist() crashes the compiler:
    // https://bugs.swift.org/browse/SR-7706
    // let batchSize = Tensor<Float>(Float(images.shape[0]))
    let batchSize = Float(images.shape[0])

    // Hyper-parameters.
    let learningRate: Float = 0.2
    var loss = Float.infinity

    // Training loop.
    print("Begin training for \(iterationCount) iterations.")

    for i in 0...iterationCount {
        // Forward pass.
        let z1 = images • parameters.w1 + parameters.b1
        let h1 = sigmoid(z1)
        let z2 = h1 • parameters.w2 + parameters.b2
        let predictions = sigmoid(z2)

        // Backward pass. This will soon be replaced by automatic
        // differentiation.
        let dz2 = (predictions - labels) / batchSize
        let dw2 = h1.transposed() • dz2
        let db2 = dz2.sum(squeezingAxes: 0)
        let dz1 = matmul(dz2, parameters.w2.transposed()) * h1 * (1 - h1)
        let dw1 = images.transposed() • dz1
        let db1 = dz1.sum(squeezingAxes: 0)
        let gradients = MNISTParameters(w1: dw1, w2: dw2, b1: db1, b2: db2)

        // Update parameters.
        parameters.update(withGradients: gradients) { param, grad in
          param -= grad * learningRate
        }

        // Update the sigmoid-based cross-entropy loss, where we treat the 10
        // class labels as independent. This is unnecessary for the MNIST case,
        // where we want to predict a single label. In that case we should
        // consider switching to a softmax-based cross-entropy loss.
        let part1 = -labels * log(predictions)
        let part2 = -(1 - labels) * log(1 - predictions)
        loss = (part1 + part2).sum() / batchSize
      
        print("Loss:", loss)
    }
}

var parameters = MNISTParameters()
// Start training.
train(&parameters, iterationCount: 20)
