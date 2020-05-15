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

import Datasets
import Foundation
import ModelSupport
import TensorFlow
import Batcher

let epochCount = 10
let batchSize = 100
let imageHeight = 28
let imageWidth = 28

let outputFolder = "./output/"
let dataset = FashionMNIST(batchSize: batchSize, device: Device.default, 
    entropy: SystemRandomNumberGenerator(), flattening: true)
// An autoencoder.
var autoencoder = Sequential {
    // The encoder.
    Dense<Float>(inputSize: imageHeight * imageWidth, outputSize: 128, activation: relu)
    Dense<Float>(inputSize: 128, outputSize: 64, activation: relu)
    Dense<Float>(inputSize: 64, outputSize: 12, activation: relu)
    Dense<Float>(inputSize: 12, outputSize: 3, activation: relu)
    // The decoder.
    Dense<Float>(inputSize: 3, outputSize: 12, activation: relu)
    Dense<Float>(inputSize: 12, outputSize: 64, activation: relu)
    Dense<Float>(inputSize: 64, outputSize: 128, activation: relu)
    Dense<Float>(inputSize: 128, outputSize: imageHeight * imageWidth, activation: tanh)
}
let optimizer = RMSProp(for: autoencoder)


// Training loop
for (epoch, epochBatches) in dataset.training.prefix(epochCount).enumerated() {
    for batch in epochBatches {
        let x = batch.data

        let 𝛁model = TensorFlow.gradient(at: autoencoder) { autoencoder -> Tensor<Float> in
            let image = autoencoder(x)
            return meanSquaredError(predicted: image, expected: x)
        }

        optimizer.update(&autoencoder, along: 𝛁model)
    }

    var testLossSum: Float = 0
    var testBatchCount = 0
    for batch in dataset.validation {
        let sampleImages = batch.data
        let testImages = autoencoder(sampleImages)

        do {
            try saveImage(
                sampleImages[0..<1], shape: (imageWidth, imageHeight), format: .grayscale,
                directory: outputFolder, name: "epoch-\(epoch)-input")
            try saveImage(
                testImages[0..<1], shape: (imageWidth, imageHeight), format: .grayscale,
                directory: outputFolder, name: "epoch-\(epoch)-output")
        } catch {
            print("Could not save image with error: \(error)")
        }

        testLossSum += meanSquaredError(predicted: testImages, expected: sampleImages).scalarized()
        testBatchCount += 1
    }
    print(
        """
        [Epoch \(epoch)] \
        Loss: \(testLossSum / Float(testBatchCount))
        """
    )
}
