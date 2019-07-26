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

// Import Python modules.
let matplotlib = Python.import("matplotlib")
let np = Python.import("numpy")
let plt = Python.import("matplotlib.pyplot")

// Turn off using display on server / Linux.
matplotlib.use("Agg")

let epochCount = 10
let batchSize = 32
let outputFolder = "./output/"
let imageHeight = 28, imageWidth = 28
let imageSize = imageHeight * imageWidth
let latentSize = 64

func plotImage(_ image: Tensor<Float>, name: String) {
    // Create figure.
    let ax = plt.gca()
    let array = np.array([image.scalars])
    let pixels = array.reshape(image.shape)
    if !FileManager.default.fileExists(atPath: outputFolder) {
        try! FileManager.default.createDirectory(
            atPath: outputFolder,
            withIntermediateDirectories: false,
            attributes: nil)
    }
    ax.imshow(pixels, cmap: "gray")
    plt.savefig("\(outputFolder)\(name).png", dpi: 300)
    plt.close()
}

/// Reads a file into an array of bytes.
func readFile(_ filename: String) -> [UInt8] {
    let possibleFolders = [".", "Resources", "GAN/Resources"]
    for folder in possibleFolders {
        let parent = URL(fileURLWithPath: folder)
        let filePath = parent.appendingPathComponent(filename).path
        guard FileManager.default.fileExists(atPath: filePath) else {
            continue
        }
        let d = Python.open(filePath, "rb").read()
        return Array(numpy: np.frombuffer(d, dtype: np.uint8))!
    }
    print("Failed to find file with name \(filename) in the following folders: \(possibleFolders).")
    exit(-1)
}

/// Reads MNIST images from specified file path.
func readMNIST(imagesFile: String) -> Tensor<Float> {
    print("Reading data.")
    let images = readFile(imagesFile).dropFirst(16).map { Float($0) }
    let rowCount = images.count / imageSize

    print("Constructing data tensors.")
    return Tensor(shape: [rowCount, imageHeight * imageWidth], scalars: images) / 255.0 * 2 - 1
}

// Models

struct Generator: Layer {
    var dense1 = Dense<Float>(inputSize: latentSize, outputSize: latentSize * 2,
                              activation: { leakyRelu($0) })
    var dense2 = Dense<Float>(inputSize: latentSize * 2, outputSize: latentSize * 4,
                              activation: { leakyRelu($0) })
    var dense3 = Dense<Float>(inputSize: latentSize * 4, outputSize: latentSize * 8,
                              activation: { leakyRelu($0) })
    var dense4 = Dense<Float>(inputSize: latentSize * 8, outputSize: imageSize,
                              activation: tanh)
    
    var batchnorm1 = BatchNorm<Float>(featureCount: latentSize * 2)
    var batchnorm2 = BatchNorm<Float>(featureCount: latentSize * 4)
    var batchnorm3 = BatchNorm<Float>(featureCount: latentSize * 8)
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let x1 = batchnorm1(dense1(input))
        let x2 = batchnorm2(dense2(x1))
        let x3 = batchnorm3(dense3(x2))
        return dense4(x3)
    }
}

struct Discriminator: Layer {
    var dense1 = Dense<Float>(inputSize: imageSize, outputSize: 256,
                              activation: { leakyRelu($0) })
    var dense2 = Dense<Float>(inputSize: 256, outputSize: 64,
                              activation: { leakyRelu($0) })
    var dense3 = Dense<Float>(inputSize: 64, outputSize: 16,
                              activation: { leakyRelu($0) })
    var dense4 = Dense<Float>(inputSize: 16, outputSize: 1,
                              activation: identity)
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        input.sequenced(through: dense1, dense2, dense3, dense4)
    }
}

// Loss functions

@differentiable
func generatorLoss(fakeLogits: Tensor<Float>) -> Tensor<Float> {
    sigmoidCrossEntropy(logits: fakeLogits,
                        labels: Tensor(ones: fakeLogits.shape))
}

@differentiable
func discriminatorLoss(realLogits: Tensor<Float>, fakeLogits: Tensor<Float>) -> Tensor<Float> {
    let realLoss = sigmoidCrossEntropy(logits: realLogits,
                                       labels: Tensor(ones: realLogits.shape))
    let fakeLoss = sigmoidCrossEntropy(logits: fakeLogits,
                                       labels: Tensor(zeros: fakeLogits.shape))
    return realLoss + fakeLoss
}

/// Returns `size` samples of noise vector.
func sampleVector(size: Int) -> Tensor<Float> {
    Tensor<Float>(randomNormal: [size, latentSize])
}

// MNIST data logic

func minibatch<Scalar>(in x: Tensor<Scalar>, at index: Int) -> Tensor<Scalar> {
    let start = index * batchSize
    return x[start..<start+batchSize]
}

let images = readMNIST(imagesFile: "train-images-idx3-ubyte")

var generator = Generator()
var discriminator = Discriminator()

let optG = Adam(for: generator, learningRate: 2e-4, beta1: 0.5)
let optD = Adam(for: discriminator, learningRate: 2e-4, beta1: 0.5)

// Noise vectors and plot function for testing
let testImageGridSize = 4
let testVector = sampleVector(size: testImageGridSize * testImageGridSize)
func plotTestImage(_ testImage: Tensor<Float>, name: String) {
    var gridImage = testImage.reshaped(to: [testImageGridSize, testImageGridSize,
                                            imageHeight, imageWidth])
    // Add padding.
    gridImage = gridImage.padded(forSizes: [(0, 0), (0, 0), (1, 1), (1, 1)], with: 1)
    // Transpose to create single image.
    gridImage = gridImage.transposed(withPermutations: [0, 2, 1, 3])
    gridImage = gridImage.reshaped(to: [(imageHeight + 2) * testImageGridSize,
                                        (imageWidth + 2) * testImageGridSize])
    // Convert [-1, 1] range to [0, 1] range.
    gridImage = (gridImage + 1) / 2
    plotImage(gridImage, name: name)
}

print("Start training...")

// Start training loop.
for epoch in 1...epochCount {
    // Start training phase.
    Context.local.learningPhase = .training
    for i in 0 ..< Int(images.shape[0]) / batchSize {
        // Perform alternative update.
        // Update generator.
        let vec1 = sampleVector(size: batchSize)
        
        let 𝛁generator = generator.gradient { generator -> Tensor<Float> in
            let fakeImages = generator(vec1)
            let fakeLogits = discriminator(fakeImages)
            let loss = generatorLoss(fakeLogits: fakeLogits)
            return loss
        }
        optG.update(&generator.allDifferentiableVariables, along: 𝛁generator)
        
        // Update discriminator.
        let realImages = minibatch(in: images, at: i)
        let vec2 = sampleVector(size: batchSize)
        let fakeImages = generator(vec2)
        
        let 𝛁discriminator = discriminator.gradient { discriminator -> Tensor<Float> in
            let realLogits = discriminator(realImages)
            let fakeLogits = discriminator(fakeImages)
            let loss = discriminatorLoss(realLogits: realLogits, fakeLogits: fakeLogits)
            return loss
        }
        optD.update(&discriminator.allDifferentiableVariables, along: 𝛁discriminator)
    }
    
    // Start inference phase.
    Context.local.learningPhase = .inference
    let testImage = generator(testVector)
    plotTestImage(testImage, name: "epoch-\(epoch)-output")
    
    let lossG = generatorLoss(fakeLogits: testImage)
    print("[Epoch: \(epoch)] Loss-G: \(lossG)")
}
