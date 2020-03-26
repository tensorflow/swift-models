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
let dataset = FashionMNIST(batchSize: batchSize, flattening: true)
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

let individualTestImages = Batcher(on: dataset.testBatcher.dataset, batchSize: 1)
var testImageIterator = individualTestImages.sequenced()

// Training loop
for epoch in 1...epochCount {
    if let nextIndividualImage = testImageIterator.next() {
        let sampleTensor = nextIndividualImage.first
        let sampleImage = Tensor(
            shape: [1, imageHeight * imageWidth], scalars: sampleTensor.scalars)

        let testImage = autoencoder(sampleImage)

        do {
            try saveImage(
                sampleImage, shape: (imageWidth, imageHeight), format: .grayscale,
                directory: outputFolder, name: "epoch-\(epoch)-input")
            try saveImage(
                testImage, shape: (imageWidth, imageHeight), format: .grayscale,
                directory: outputFolder, name: "epoch-\(epoch)-output")
        } catch {
            print("Could not save image with error: \(error)")
        }

        let sampleLoss = meanSquaredError(predicted: testImage, expected: sampleImage)
        print("[Epoch: \(epoch)] Loss: \(sampleLoss)")
    }

    for batch in dataset.trainingBatcher.sequenced() {
        let x = batch.first

        let 𝛁model = TensorFlow.gradient(at: autoencoder) { autoencoder -> Tensor<Float> in
            let image = autoencoder(x)
            return meanSquaredError(predicted: image, expected: x)
        }

        optimizer.update(&autoencoder, along: 𝛁model)
    }
}
