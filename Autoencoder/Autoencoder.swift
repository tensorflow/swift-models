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
import Python

// Import Python modules
let matplotlib = Python.import("matplotlib")
let np = Python.import("numpy")
let plt = Python.import("matplotlib.pyplot")
// Turn off using display on server / linux
matplotlib.use("Agg")

let outputFolder = "/tmp/mnist-test/"

func readDataset() -> (images: Tensor<Float>, labels: Tensor<Int32>) {
    print("Reading the data.")
    let swiftFileURL = URL(fileURLWithPath: #file)
    var imageFolderURL = swiftFileURL.deletingLastPathComponent()
    var labelFolderURL = swiftFileURL.deletingLastPathComponent()
    imageFolderURL.appendPathComponent("Resources/train-images-idx3-ubyte")
    labelFolderURL.appendPathComponent("Resources/train-labels-idx1-ubyte")

    guard let imageData = try? Data(contentsOf: imageFolderURL).dropFirst(16),
          let labelData = try? Data(contentsOf: labelFolderURL).dropFirst(8) else {
        print("Error: could not read dataset.")
        exit(-1)
    }
    let images = imageData.map { Float($0) }
    let labels = labelData.map { Int32($0) }
    let rowCount = Int32(labels.count)
    let columnCount = Int32(images.count) / rowCount

    print("Constructing the data tensors.")
    let imagesTensor = Tensor(shape: [rowCount, columnCount], scalars: images).toAccelerator() / 255.0
    let labelsTensor = Tensor(labels).toAccelerator()
    return (imagesTensor, labelsTensor)
}

func plot(image: [Float], labels: Tensor<Int32>, step: Int) {
    // Create figure
    let ax = plt.gca()
    // Create colors
    let linspace = np.linspace(start: 0, stop: 1, num: 10, endpoint: true, retstep: false)
    let colorMap = plt.get_cmap("RdBu")
    let colors = np.r_[linspace, linspace]
    let mappedColors = colorMap(colors)
    for element in 0..<10 {
        let xElementIndex = labels.scalars.enumerated().filter { $0.element == element }.map { 2 * $0.offset + 0 }
        let yElementIndex = labels.scalars.enumerated().filter { $0.element == element }.map { 2 * $0.offset + 1 }
        let x = image.enumerated().filter { xElementIndex.contains($0.offset) }.map { $0.element }
        let y = image.enumerated().filter { yElementIndex.contains($0.offset) }.map { $0.element }
        ax.scatter(x: x, y: y, color: mappedColors[element], label: "\(element)", s: 20, alpha: 0.8)
    }
    ax.set_xlabel("Autoencoder clasterization.")
    ax.legend()
    if !FileManager.default.fileExists(atPath: outputFolder) {
        try! FileManager.default.createDirectory(atPath: outputFolder, withIntermediateDirectories: false, attributes: nil)
    }
    plt.savefig("\(outputFolder)autoencoder-\(step).png", dpi: 300)
    plt.close()
}

func plot(image: [Float], name: String) {
    // Create figure
    let ax = plt.gca()
    let array = np.array([image])
    let pixels = array.reshape([Autoencoder.imageEdge * 5, Autoencoder.imageEdge * 5])
    if !FileManager.default.fileExists(atPath: outputFolder) {
        try! FileManager.default.createDirectory(atPath: outputFolder, withIntermediateDirectories: false, attributes: nil)
    }
    ax.imshow(pixels, cmap: "gray")
    plt.savefig("\(outputFolder)\(name).png", dpi: 300)
    plt.close()
}

struct Autoencoder : Parameterized {
    static let imageEdge: Int32 = 28
    static let imageSize: Int32 = imageEdge * imageEdge
    static let decoderLayerSize: Int32 = 50
    static let encoderLayerSize: Int32 = 50
    static let hiddenLayerSize: Int32 = 2

    @TFParameter var w1: Tensor<Float>
    @TFParameter var w2: Tensor<Float>
    @TFParameter var w3: Tensor<Float>
    @TFParameter var w4: Tensor<Float>
    @TFParameter var b2 = Tensor<Float>(zeros: [1, Autoencoder.hiddenLayerSize])

    var learningRate: Float = 0.001

    init() {
        let w1 = Tensor<Float>(randomUniform: [Autoencoder.imageSize, Autoencoder.decoderLayerSize])
        let w2 = Tensor<Float>(randomUniform: [Autoencoder.decoderLayerSize, Autoencoder.hiddenLayerSize])
        let w3 = Tensor<Float>(randomUniform: [Autoencoder.hiddenLayerSize, Autoencoder.encoderLayerSize])
        let w4 = Tensor<Float>(randomUniform: [Autoencoder.encoderLayerSize, Autoencoder.imageSize])

        // Xavier initialization
        self.w1 = w1 / sqrtf(Float(Autoencoder.imageSize))
        self.w2 = w2 / sqrtf(Float(Autoencoder.decoderLayerSize))
        self.w3 = w3 / sqrtf(Float(Autoencoder.hiddenLayerSize))
        self.w4 = w4 / sqrtf(Float(Autoencoder.encoderLayerSize))
    }
}

extension Autoencoder {
    func embedding(for input: Tensor<Float>) -> (tensor: Tensor<Float>, loss: Float, input: Tensor<Float>, output: Tensor<Float>) {
        // Forward pass
        let z1 = input • w1
        let h1 = tanh(z1)
        let z2 = h1 • w2 + b2
        let h2 = z2
        let z3 = h2 • w3
        let h3 = tanh(z3)
        let z4 = h3 • w4
        let predictions = sigmoid(z4)
        let loss: Float = 0.5 * (predictions - input).squared().mean()
        return (h2, loss, input, predictions)
    }

    mutating func trainStep(input: Tensor<Float>) -> Float {
        let learningRate = self.learningRate

        // Batch normalization
        let batchSize = Tensor<Float>(input.shapeTensor[0])

        // Forward pass
        let z1 = input • w1
        let h1 = tanh(z1)
        let z2 = h1 • w2 + b2
        let h2 = z2
        let z3 = h2 • w3
        let h3 = tanh(z3)
        let z4 = h3 • w4
        let predictions = sigmoid(z4)

        // Backward pass
        let dz4 = (predictions - input) / batchSize
        let dw4 = h3.transposed() • dz4
        let dz3 = matmul(dz4, w4.transposed()) * (1 - h3.squared())
        let dw3 = h2.transposed() • dz3
        let dz2 = matmul(dz3, w3.transposed())
        let dw2 = h1.transposed() • dz2
        let db2 = dz2.sum(squeezingAxes: 0)
        let dz1 = matmul(dz2, w2.transposed()) * (1 - h1.squared())
        let dw1 = input.transposed() • dz1
        let gradients = Parameters(w1: dw1, w2: dw2, w3: dw3, w4: dw4, b2: db2)

        let loss: Float = 0.5 * (predictions - input).squared().mean()

        // Gradient descent.
        allParameters.update(withGradients: gradients) { p, g in
            p -= g * learningRate
        }

        return loss
    }
}

extension Autoencoder {
    mutating public func train(on dataset: (images: Tensor<Float>, labels: Tensor<Int32>),
                               iterationCount: Int) {
        print("Train on dataset")
        let batchSize: Int32 = 50
        for i in 1...iterationCount {
            for batchStep in 0..<500 {
                let batch = batchSize * Int32(batchStep)
                let images = dataset.images.slice(lowerBounds: [batch, 0], upperBounds: [batch + batchSize, Autoencoder.imageSize])
                let loss = trainStep(input: images)
                if i % 10 == 0 && batchStep == 0 {
                    print("\(i) steps, loss: \(loss)")
                }
            }
        }
    }

    private static func reshape(image: [Float], imageCountPerLine: Int) -> [Float] {
        var fullImage: [Float] = []
        let imageEdge = Int(Autoencoder.imageEdge)

        // FIXME: Improve for's.
        for rowIndex in 0..<imageCountPerLine {
            for pixelIndex in 0..<imageEdge {
                for imageIndex in 0..<imageCountPerLine {
                    let rowShift: Int = rowIndex * Int(imageSize) * imageCountPerLine
                    let startIndex = rowShift + ((imageIndex * imageEdge) + pixelIndex) * imageEdge
                    let endIndex = startIndex + imageEdge
                    fullImage.append(contentsOf: image[startIndex..<endIndex])
                }
            }
        }
        return fullImage
    }

    @inline(never)
    func embedding(from dataset: (images: Tensor<Float>, labels: Tensor<Int32>), shouldSaveInput: Bool, elementCount: Int32, step: Int) -> (labels: Tensor<Int32>, tensor: [Float]) {
        let images = dataset.images.slice(lowerBounds: [0, 0], upperBounds: [elementCount, Autoencoder.imageSize])
        let labels = dataset.labels.slice(lowerBounds: [0], upperBounds: [elementCount])
        let result = embedding(for: images)
        print("Embedding loss: ", result.loss)
        let size = Int(Autoencoder.imageSize)
        let imagesInLine = 5
        if shouldSaveInput {
            let inputImage = Array(result.input.scalars[0..<imagesInLine * imagesInLine * size])
            plot(image: Autoencoder.reshape(image: inputImage, imageCountPerLine: imagesInLine), name: "input-\(step)")
        }
        let outputImage = Array(result.output.scalars[0..<imagesInLine * imagesInLine * size])
        plot(image: Autoencoder.reshape(image: outputImage, imageCountPerLine: imagesInLine), name: "output-\(step)")
        return (labels: labels, tensor: result.tensor.scalars)
    }
}

let dataset = readDataset()
var autoencoder = Autoencoder()

// Initial prediction.
var embedding = autoencoder.embedding(from: dataset, shouldSaveInput: true, elementCount: 300, step: 0)
plot(image: embedding.tensor, labels: embedding.labels, step: 0)

for i in 1...5 {
    autoencoder.train(on: dataset, iterationCount: 100)
    embedding = autoencoder.embedding(from: dataset, shouldSaveInput: false, elementCount: 300, step: i)
    plot(image: embedding.tensor, labels: embedding.labels, step: i)
}

print("Autoencoder results saved to \(outputFolder).")
