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

/// Based on https://blog.keras.io/building-autoencoders-in-keras.html

import Datasets
import Foundation
import ModelSupport
import TensorFlow

let epochCount = 10
let batchSize = 100
let imageHeight = 28
let imageWidth = 28

let outputFolder = "./output/"
let dataset = KuzushijiMNIST(flattening: true)

// An autoencoder.
struct Autoencoder2D: Layer {
    var encoder1 = Conv2D<Float>(filterShape: (3, 3, 1, 16), padding: .same, activation: relu)
    var encoder2 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2), padding: .same)
    var encoder3 = Conv2D<Float>(filterShape: (3, 3, 16, 8), padding: .same, activation: relu)
    var encoder4 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2), padding: .same)
    var encoder5 = Conv2D<Float>(filterShape: (3, 3, 8, 8), padding: .same, activation: relu)
    var encoder6 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2), padding: .same)

    var decoder1 = Conv2D<Float>(filterShape: (3, 3, 8, 8), padding: .same, activation: relu)
    var decoder2 = UpSampling2D<Float>(size: 2)
    var decoder3 = Conv2D<Float>(filterShape: (3, 3, 8, 8), padding: .same, activation: relu)
    var decoder4 = UpSampling2D<Float>(size: 2)
    var decoder5 = Conv2D<Float>(filterShape: (3, 3, 8, 16), activation: relu)
    var decoder6 = UpSampling2D<Float>(size: 2)

    var output = Conv2D<Float>(filterShape: (3, 3, 16, 1), padding: .same, activation: sigmoid)

    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let resize = input.reshaped(to: [localBatchSize, 28, 28, 1])
        let encoder = resize.sequenced(through: encoder1,
            encoder2, encoder3, encoder4, encoder5, encoder6)
        let decoder = encoder.sequenced(through: decoder1,
            decoder2, decoder3, decoder4, decoder5, decoder6)
        return output(decoder).reshaped(to: [localBatchSize, imageHeight * imageWidth])
    }
}

var model = Autoencoder2D()
let optimizer = AdaDelta(for: model)

let individualTestImages = dataset.testDataset.batched(1)
var testImageIterator = individualTestImages.makeIterator()
var localBatchSize = batchSize

// Training loop
for epoch in 1...epochCount {
    localBatchSize = 1
    if let nextIndividualImage = testImageIterator.next() {
        let sampleTensor = nextIndividualImage.data
        let sampleImage = Tensor(
            shape: [28, 28], scalars: sampleTensor.scalars)

        let testImage = model(sampleImage).reshaped(to: [localBatchSize, 28, 28, 1])

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

    localBatchSize = batchSize
    let trainingShuffled = dataset.trainingDataset.shuffled(
        sampleCount: dataset.trainingExampleCount, randomSeed: Int64(epoch))
    for batch in trainingShuffled.batched(batchSize) {
        let x = batch.data

        let 𝛁model = TensorFlow.gradient(at: model) { model -> Tensor<Float> in
            let image = model(x)
            return meanSquaredError(predicted: image, expected: x)
        }

        optimizer.update(&model, along: 𝛁model)
    }
}
