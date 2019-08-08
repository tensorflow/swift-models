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
import Python
import Datasets

// Import Python modules
let matplotlib = Python.import("matplotlib")
let np = Python.import("numpy")

// Use the AGG renderer for saving images to disk.
matplotlib.use("Agg")

let plt = Python.import("matplotlib.pyplot")

let epochCount = 10
let batchSize = 100
let outputFolder = "./output/"
let imageHeight = 28, imageWidth = 28

func plot(image: [Float], name: String) {
    // Create figure
    let ax = plt.gca()
    let array = np.array([image])
    let pixels = array.reshape([imageHeight, imageWidth])
    if !FileManager.default.fileExists(atPath: outputFolder) {
        try! FileManager.default.createDirectory(atPath: outputFolder,
                            withIntermediateDirectories: false,
                                             attributes: nil)
    }
    ax.imshow(pixels, cmap: "gray")
    plt.savefig("\(outputFolder)\(name).png", dpi: 300)
    plt.close()
}

/// An autoencoder.
struct Autoencoder: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var encoder1 = Dense<Float>(inputSize: imageHeight * imageWidth, outputSize: 128,
        activation: relu)
    var encoder2 = Dense<Float>(inputSize: 128, outputSize: 64, activation: relu)
    var encoder3 = Dense<Float>(inputSize: 64, outputSize: 12, activation: relu)
    var encoder4 = Dense<Float>(inputSize: 12, outputSize: 3, activation: relu)

    var decoder1 = Dense<Float>(inputSize: 3, outputSize: 12, activation: relu)
    var decoder2 = Dense<Float>(inputSize: 12, outputSize: 64, activation: relu)
    var decoder3 = Dense<Float>(inputSize: 64, outputSize: 128, activation: relu)
    var decoder4 = Dense<Float>(inputSize: 128, outputSize: imageHeight * imageWidth,
        activation: tanh)

    @differentiable
    func callAsFunction(_ input: Input) -> Output {
        let encoder = input.sequenced(through: encoder1, encoder2, encoder3, encoder4)
        return encoder.sequenced(through: decoder1, decoder2, decoder3, decoder4)
    }
}

let dataset = MNIST(batchSize: batchSize, flattening: true)
var autoencoder = Autoencoder()
let optimizer = RMSProp(for: autoencoder)

// Training loop
for epoch in 1...epochCount {
    let sampleImage = Tensor(shape: [1, imageHeight * imageWidth], scalars: dataset.trainingImages[epoch].scalars)
    let testImage = autoencoder(sampleImage)

    plot(image: sampleImage.scalars, name: "epoch-\(epoch)-input")
    plot(image: testImage.scalars, name: "epoch-\(epoch)-output")

    let sampleLoss = meanSquaredError(predicted: testImage, expected: sampleImage)
    print("[Epoch: \(epoch)] Loss: \(sampleLoss)")

    for i in 0 ..< dataset.trainingSize / batchSize {
        let x = dataset.trainingImages.minibatch(at: i, batchSize: batchSize)

        let 𝛁model = autoencoder.gradient { autoencoder -> Tensor<Float> in
            let image = autoencoder(x)
            return meanSquaredError(predicted: image, expected: x)
        }

        optimizer.update(&autoencoder, along: 𝛁model)
    }
}
