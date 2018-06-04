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

public enum AutoencoderError: Error {
    case noDatasetFound
}

public func readDataset() throws -> (images: Tensor<Float>, labels: Tensor<Int32>) {
    print("Reading the data.")
    guard let swiftFile = CommandLine.arguments.first else { throw AutoencoderError.noDatasetFound }
    let swiftFileURL = URL(fileURLWithPath: swiftFile)
    var imageFolderURL = swiftFileURL.deletingLastPathComponent()
    var labelFolderURL = swiftFileURL.deletingLastPathComponent()
    imageFolderURL.appendPathComponent("Resources/train-images-idx3-ubyte")
    labelFolderURL.appendPathComponent("Resources/train-labels-idx1-ubyte")

    let imageData = try Data(contentsOf: imageFolderURL).dropFirst(16)
    let labelData = try Data(contentsOf: labelFolderURL).dropFirst(8)
    let images = imageData.map { Float($0) }
    let labels = labelData.map { Int32($0) }
    let rowCount = Int32(labels.count)
    let columnCount = Int32(images.count) / rowCount
    
    print("Constructing the data tensors.")
    let imagesTensor = Tensor(shape: [rowCount, columnCount], scalars: images)
    let labelsTensor = Tensor(labels)
    return (imagesTensor, labelsTensor)
}

public func plot(tensor: [Float], labels: Tensor<Int32>, step: Int) {
    // Import Python modules
    let matplotlib = Python.import("matplotlib")
    // Turn off using display on server / linux
    matplotlib.use("Agg")

    let np = Python.import("numpy")
    let cm = Python.import("matplotlib.cm")
    let plt = Python.import("matplotlib.pyplot")
   
    // Create figure
    let ax = plt.gca()
    // Create colors
    let linspace = np.linspace(start: 0, stop: 1, num: 10, endpoint: true, retstep: false)
    let rainbow = cm.rainbow([linspace])
    for element in 0..<10 {
        let xElementIndex = labels.scalars.enumerated().filter { $0.element == element}.map { 2 * $0.offset + 0}
        let yElementIndex = labels.scalars.enumerated().filter { $0.element == element}.map { 2 * $0.offset + 1}
        let x = tensor.enumerated().filter { xElementIndex.contains($0.offset) }.map { $0.element }
        let y = tensor.enumerated().filter { yElementIndex.contains($0.offset) }.map { $0.element }
        ax.scatter(x: x, y: y, color: rainbow[0][element], label: "\(element)", s: 20, alpha: 0.8)
    }
    ax.set_xlabel(["x label"])
    ax.set_ylabel(["y label"])
    ax.legend()
    //plt.axis.dynamicallyCall(withArguments:[-0.3, 0.3, -0.3, 0.3])
    if !FileManager.default.fileExists(atPath: "/tmp/mnist-test/") {
        try? FileManager.default.createDirectory(atPath: "/tmp/mnist-test/", withIntermediateDirectories: false, attributes: nil)
    }
    plt.savefig("/tmp/mnist-test/autoencoder-\(step).png", dpi: 300)
    plt.close()
}

public func plot(image: [Float], name: String) {
    // Import Python modules
    let matplotlib = Python.import("matplotlib")
    // Turn off using display on server / linux
    matplotlib.use("Agg")
    let np = Python.import("numpy")
    let plt = Python.import("matplotlib.pyplot")
    // Create figure
    let ax = plt.gca()

    // Create colors
    let array = np.array([image])

    let pixels = array.reshape([Autoencoder.imageEdge, Autoencoder.imageEdge])
    if !FileManager.default.fileExists(atPath: "/tmp/mnist-test/") {
        try? FileManager.default.createDirectory(atPath: "/tmp/mnist-test/", withIntermediateDirectories: false, attributes: nil)
    }
    ax.imshow(pixels, cmap: "gray")
    plt.savefig("/tmp/mnist-test/\(name).png", dpi : 300)
    plt.close()
}

public struct Autoencoder {
    var w1 = Tensor<Float>(randomUniform: [Autoencoder.imageSize, 50])
    var w2 = Tensor<Float>(randomUniform: [50, 2])
    var w3 = Tensor<Float>(randomUniform: [2, 50])
    var w4 = Tensor<Float>(randomUniform: [50, Autoencoder.imageSize])
    var learningRate: Float = 0.001
}

public extension Autoencoder {
    static let imageEdge: Int32 = 28
    static let imageSize: Int32 = imageEdge * imageEdge
    
    @inline(never)
    func embedding(input: Tensor<Float>) -> (tensor: [Float], loss: Float, input: [Float], output: [Float]) {
        let inputNormalized = input / 255.0

        // Forward pass
        let z1 = inputNormalized ⊗ self.w1
        let h1 = tanh(z1)
        let z2 = h1 ⊗ self.w2
        let h2 = tanh(z2)
        let z3 = h2 ⊗ self.w3
        let h3 = tanh(z3)
        let z4 = h3 ⊗ self.w4
        let predictions = sigmoid(z4)
        let loss: Float = 0.5 * (predictions - inputNormalized).squared().mean()
        return (self.w2.scalars, loss, inputNormalized.scalars, predictions.scalars)
    }
    
    mutating func trainStep(input: Tensor<Float>, applyUpdate: Bool = true) -> Float  {
        var w1 = self.w1
        var w2 = self.w2
        var w3 = self.w3
        var w4 = self.w4
        let learningRate = self.learningRate

        // Batch normalization
        let inputNormalized = input / 255.0
        let batchSize = Tensor<Float>(inputNormalized.shapeTensor[0])

        // Forward pass
        let z1 = inputNormalized ⊗ w1
        let h1 = tanh(z1)
        let z2 = h1 ⊗ w2
        let h2 = tanh(z2)
        let z3 = h2 ⊗ w3
        let h3 = tanh(z3)
        let z4 = h3 ⊗ w4
        let predictions = sigmoid(z4)

        // Backward pass
        //FIXME: Add sigmoid derivative
        let dz4 = ((predictions - inputNormalized) / batchSize)
        let dw4 = h3.transposed(withPermutations: 1, 0) ⊗ dz4
        let dz3 = dz4.dot(w4.transposed(withPermutations: 1, 0)) * (1 - h3.squared())
        let dw3 = h2.transposed(withPermutations: 1, 0) ⊗ dz3
        let dz2 = dz3.dot(w3.transposed(withPermutations: 1, 0)) * (1 - h2.squared())
        let dw2 = h1.transposed(withPermutations: 1, 0) ⊗ dz2
        let dz1 = dz2.dot(w2.transposed(withPermutations: 1, 0)) * (1 - h1.squared())
        let dw1 = inputNormalized.transposed(withPermutations: 1, 0) ⊗ dz1

        let loss: Float = 0.5 * (predictions - inputNormalized).squared().mean()

        // Gradient descent.
        w1 -= dw1 * learningRate
        w2 -= dw2 * learningRate
        w3 -= dw3 * learningRate
        w4 -= dw4 * learningRate
        
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        
        return loss
    }
}

public extension Autoencoder {
    @inline(never)
    mutating public func train(on dataset: (images: Tensor<Float>, labels: Tensor<Int32>)) {
        print("Train on dataset")
        var iterationNumber = 0
        let maxIterations = 10
        let batchSize: Int32 = 30
        repeat {
            iterationNumber += 1
            for batchStep in 0..<1000 {
                let batch = batchSize * Int32(batchStep)
                let images = dataset.images.slice(lowerBounds: [batch, 0], upperBounds: [batch + batchSize, Autoencoder.imageSize])
                
                let loss = trainStep(input: images)
                if iterationNumber % 2 == 0 && batchStep == 0 {
                    print("\(iterationNumber) step, loss: \(loss)")
                }
            }
        } while iterationNumber < maxIterations
    }
    
    func embedding(from dataset: (images: Tensor<Float>, labels: Tensor<Int32>), saveInput: Bool, numberOfElements: Int32, step: Int) -> (labels: Tensor<Int32>, tensor: [Float]) {
        let images = dataset.images.slice(lowerBounds: [0, 0], upperBounds: [numberOfElements, Autoencoder.imageSize])
        let labels = dataset.labels.slice(lowerBounds: [0], upperBounds: [numberOfElements])
        let result = embedding(input: images)
        print("Embedding loss: ", result.loss)        
        let size = Int(Autoencoder.imageSize)
        
        for stepIndex in 0..<10 {
            if saveInput {
                plot(image: Array(Array<Float>(result.input)[(stepIndex * size)..<(stepIndex + 1) * size]), name: "input-\(step)-\(stepIndex)")
            }
            plot(image: Array(Array<Float>(result.output)[(stepIndex * size)..<(stepIndex + 1) * size]), name: "output-\(step)-\(stepIndex)")
        }
        // Some ML debug
        // print("Input range: ", result.input.max()!, result.input.min()!)
        // print("Output range: ", result.output.max()!, result.output.min()!)
        return (labels: labels, tensor: result.tensor)
    }
}


public func main() {
    do {
        let dataset = try readDataset()
        var autoencoder = Autoencoder()

        var embedding = autoencoder.embedding(from: dataset, saveInput: true, numberOfElements: 100, step:0)
        plot(tensor: embedding.tensor, labels: embedding.labels, step:0)
        
        autoencoder.train(on: dataset)
        embedding = autoencoder.embedding(from: dataset, saveInput: false, numberOfElements: 100, step:1)
        plot(tensor: embedding.tensor, labels: embedding.labels, step:1)

        autoencoder.train(on: dataset)
        embedding = autoencoder.embedding(from: dataset, saveInput: false, numberOfElements: 100, step:2)
        plot(tensor: embedding.tensor, labels: embedding.labels, step:2)

        autoencoder.train(on: dataset)
        embedding = autoencoder.embedding(from: dataset, saveInput: false, numberOfElements: 100, step:3)
        plot(tensor: embedding.tensor, labels: embedding.labels, step:3)

        autoencoder.train(on: dataset)
        embedding = autoencoder.embedding(from: dataset, saveInput: false, numberOfElements: 100, step:4)
        plot(tensor: embedding.tensor, labels: embedding.labels, step:4)

        autoencoder.train(on: dataset)
        embedding = autoencoder.embedding(from: dataset, saveInput: false, numberOfElements: 100, step:5)
        plot(tensor: embedding.tensor, labels: embedding.labels, step:5)
        
        print("Now, you can open /tmp/mnist-test/ folder and review the resolts.")
    } catch {
        print(error)
    }
}

main()
