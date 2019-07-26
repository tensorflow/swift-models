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
let batchSize = 32
let outputFolder = "./output/"
let imageHeight = 28, imageWidth = 28
let imageDim = imageHeight*imageWidth
let latentDim = 64

func plot(image: Tensor<Float>, name: String) {
    // Create figure
    let ax = plt.gca()
    let array = np.array([image.scalars])
    let pixels = array.reshape(image.shape)
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

/// Reads MNIST images and labels from specified file paths.
func readMNIST(imagesFile: String, labelsFile: String) -> (images: Tensor<Float>,
                                                           labels: Tensor<Int32>) {
    print("Reading data.")
    let images = readFile(imagesFile).dropFirst(16).map { Float($0) }
    let labels = readFile(labelsFile).dropFirst(8).map { Int32($0) }
    let rowCount = labels.count

    print("Constructing data tensors.")
    return (
        images: Tensor(shape: [rowCount, imageHeight * imageWidth], scalars: images) / 255.0 * 2 - 1,
        labels: Tensor(labels)
    )
}

// Models
struct Generator: Layer {
    var dense1 = Dense<Float>(inputSize: latentDim, outputSize: latentDim*2, activation: { leakyRelu($0) })
    var dense2 = Dense<Float>(inputSize: latentDim*2, outputSize: latentDim*4, activation: { leakyRelu($0) })
    var dense3 = Dense<Float>(inputSize: latentDim*4, outputSize: latentDim*8, activation: { leakyRelu($0) })
    var dense4 = Dense<Float>(inputSize: latentDim*8, outputSize: imageDim, activation: tanh)
    
    var batchnorm1 = BatchNorm<Float>(featureCount: latentDim*2)
    var batchnorm2 = BatchNorm<Float>(featureCount: latentDim*4)
    var batchnorm3 = BatchNorm<Float>(featureCount: latentDim*8)
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let x1 = batchnorm1(dense1(input))
        let x2 = batchnorm2(dense2(x1))
        let x3 = batchnorm3(dense3(x2))
        
        return dense4(x3)
    }
}

struct Discriminator: Layer {
    var dense1 = Dense<Float>(inputSize: imageDim, outputSize: 256, activation: { leakyRelu($0) })
    var dense2 = Dense<Float>(inputSize: 256, outputSize: 64, activation: { leakyRelu($0) })
    var dense3 = Dense<Float>(inputSize: 64, outputSize: 16, activation: { leakyRelu($0) })
    var dense4 = Dense<Float>(inputSize: 16, outputSize: 1, activation: identity)
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        input.sequenced(through: dense1, dense2, dense3, dense4)
    }
}

// Loss functions
@differentiable
func generatorLossFunc(fakeLogits: Tensor<Float>) -> Tensor<Float> {
    sigmoidCrossEntropy(logits: fakeLogits,
                        labels: Tensor(ones: [fakeLogits.shape[0], 1]))
}

@differentiable
func discriminatorLossFunc(realLogits: Tensor<Float>, fakeLogits: Tensor<Float>) -> Tensor<Float> {
    let realLoss = sigmoidCrossEntropy(logits: realLogits,
                                       labels: Tensor(ones: [realLogits.shape[0], 1]))
    let fakeLoss = sigmoidCrossEntropy(logits: fakeLogits,
                                       labels: Tensor(zeros: [fakeLogits.shape[0], 1]))
    return realLoss + fakeLoss
}

func sampleVector(size: Int) -> Tensor<Float> {
    Tensor<Float>(randomNormal: [size, latentDim])
}

// MNIST data logic
func minibatch<Scalar>(in x: Tensor<Scalar>, at index: Int) -> Tensor<Scalar> {
    let start = index * batchSize
    return x[start..<start+batchSize]
}

let (images, numericLabels) = readMNIST(imagesFile: "train-images-idx3-ubyte",
                                        labelsFile: "train-labels-idx1-ubyte")
let labels = Tensor<Float>(oneHotAtIndices: numericLabels, depth: 10)

var generator = Generator()
var discriminator = Discriminator()

let optG = Adam(for: generator, learningRate: 2e-4, beta1: 0.5)
let optD = Adam(for: discriminator, learningRate: 2e-4, beta1: 0.5)

// noise for testing and plot function
let testImageGridSize = 4
let testVector = sampleVector(size: testImageGridSize*testImageGridSize)
func plotTestImage(_ testImage: Tensor<Float>, name: String) {
    var imageGrid = testImage.reshaped(to: [testImageGridSize, testImageGridSize, imageHeight, imageWidth])
    
    // Add padding
    imageGrid = imageGrid.padded(forSizes: [(0, 0), (0, 0), (1, 1), (1, 1)], with: 1)
    
    // Transpose to create single image.
    imageGrid = imageGrid.transposed(withPermutations: [0, 2, 1, 3])
    imageGrid = imageGrid.reshaped(to: [(imageHeight+2)*testImageGridSize, (imageWidth+2)*testImageGridSize])
    
    // [-1, 1] range to [0, 1] range
    imageGrid = (imageGrid + 1) / 2
    
    plot(image: imageGrid, name: name)
}

print("Start training...")

// Training loop
for epoch in 1...epochCount {
    // Training phase
    Context.local.learningPhase = .training
    for i in 0 ..< Int(labels.shape[0]) / batchSize {
        // Alternative update
        
        // Update Generator
        let vec1 = sampleVector(size: batchSize)
        
        let 𝛁generator = generator.gradient { generator -> Tensor<Float> in
            let fakeImages = generator(vec1)
            let fakeLogits = discriminator(fakeImages)
            let loss = generatorLossFunc(fakeLogits: fakeLogits)
            return loss
        }
        optG.update(&generator.allDifferentiableVariables, along: 𝛁generator)
        
        // Update Discriminator
        let realImages = minibatch(in: images, at: i)
        let vec2 = sampleVector(size: batchSize)
        let fakeImages = generator(vec2)
        
        let 𝛁discriminator = discriminator.gradient { discriminator -> Tensor<Float> in
            let realLogits = discriminator(realImages)
            let fakeLogits = discriminator(fakeImages)
            let loss = discriminatorLossFunc(realLogits: realLogits, fakeLogits: fakeLogits)
            return loss
        }
        
        optD.update(&discriminator.allDifferentiableVariables, along: 𝛁discriminator)
    }
    
    // Inference phase
    Context.local.learningPhase = .inference
    let testImage: Tensor<Float> = generator(testVector)
    
    plotTestImage(testImage, name: "epoch-\(epoch)-output")
    
    let lossG = generatorLossFunc(fakeLogits: testImage)
    
    print("[Epoch: \(epoch)] Loss-G: \(lossG)")
}
