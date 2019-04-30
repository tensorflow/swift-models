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

// Import Python modules
let matplotlib = Python.import("matplotlib")
let np = Python.import("numpy")
let plt = Python.import("matplotlib.pyplot")

// Turn off using display on server / linux
matplotlib.use("Agg")

// Some globals
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

/// Reads a file into an array of bytes.
func readFile(_ filename: String) -> [UInt8] {
    let possibleFolders = [".", "Resources", "Autoencoder/Resources"]
    for folder in possibleFolders {
        let parent = URL(fileURLWithPath: folder)
        let filePath = parent.appendingPathComponent(filename).path
        guard FileManager.default.fileExists(atPath: filePath) else {
            continue
        }
        let d = Python.open(filePath, "rb").read()
        return Array(numpy: np.frombuffer(d, dtype: np.uint8))!
    }
    fatalError(
        "Failed to find file with name \(filename) in the following folders: \(possibleFolders).")
}

/// Reads MNIST images and labels from specified file paths.
func readMNIST(imagesFile: String, labelsFile: String) -> (images: Tensor<Float>,
                                                           labels: Tensor<Int32>) {
    print("Reading data.")
    let images = readFile(imagesFile).dropFirst(16).map { Float($0) }
    let labels = readFile(labelsFile).dropFirst(8).map { Int32($0) }
    let rowCount = labels.count

    print("Constructing data tensors.")
    return (
        images: Tensor(shape: [rowCount, imageHeight * imageWidth], scalars: images) / 255.0,
        labels: Tensor(labels)
    )
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
    func call(_ input: Input) -> Output {
        let encoder = input.sequenced(through: encoder1, encoder2, encoder3, encoder4)
        return encoder.sequenced(through: decoder1, decoder2, decoder3, decoder4)
    }
}

// MNIST data logic
func minibatch<Scalar>(in x: Tensor<Scalar>, at index: Int) -> Tensor<Scalar> {
    let start = index * batchSize
    return x[start..<start+batchSize]
}

let (images, numericLabels) = readMNIST(imagesFile: "train-images-idx3-ubyte",
                                        labelsFile: "train-labels-idx1-ubyte")
let labels = Tensor<Float>(oneHotAtIndices: numericLabels, depth: 10)

var autoencoder = Autoencoder()
let optimizer = RMSProp(for: autoencoder)

// Training loop
for epoch in 1...epochCount {
    let sampleImage = Tensor(shape: [1, imageHeight * imageWidth], scalars: images[epoch].scalars)
    let testImage = autoencoder(sampleImage)

    plot(image: sampleImage.scalars, name: "epoch-\(epoch)-input")
    plot(image: testImage.scalars, name: "epoch-\(epoch)-output")

    let sampleLoss = meanSquaredError(predicted: testImage, expected: sampleImage)
    print("[Epoch: \(epoch)] Loss: \(sampleLoss)")

    for i in 0 ..< Int(labels.shape[0]) / batchSize {
        let x = minibatch(in: images, at: i)

        let 𝛁model = autoencoder.gradient { autoencoder -> Tensor<Float> in
            let image = autoencoder(x)
            return meanSquaredError(predicted: image, expected: x)
        }

        optimizer.update(&autoencoder.allDifferentiableVariables, along: 𝛁model)
    }
}
