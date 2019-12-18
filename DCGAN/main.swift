import Datasets
import Foundation
import TensorFlow

import Python
let plt = Python.import("matplotlib.pyplot")
let np = Python.import("numpy")

let batchSize = 512
let mnist = MNIST(batchSize: batchSize, flattening: false, normalizing: true)

let zDim = 100

// MARK: - Models

// MARK: Generator

struct Generator: Layer {
    var flatten = Flatten<Float>()

    var dense1 = Dense<Float>(inputSize: zDim, outputSize: 7 * 7 * 256)
    var batchNorm1 = BatchNorm<Float>(featureCount: 7 * 7 * 256)
    // leaky relu
    // reshape

    var transConv2D1 = TransposedConv2D<Float>(
        filterShape: (5, 5, 128, 256),
        strides: (1, 1),
        padding: .same
    )
    // flatten
    var batchNorm2 = BatchNorm<Float>(featureCount: 7 * 7 * 128)
    // leaky relu

    var transConv2D2 = TransposedConv2D<Float>(
        filterShape: (5, 5, 64, 128),
        strides: (2, 2),
        padding: .same
    )
    // flatten
    var batchNorm3 = BatchNorm<Float>(featureCount: 14 * 14 * 64)
    // leaky relu

    var transConv2D3 = TransposedConv2D<Float>(
        filterShape: (5, 5, 1, 64),
        strides: (2, 2),
        padding: .same
    )
    // tanh

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let x1 = leakyRelu(input.sequenced(through: dense1, batchNorm1))
        let x1Reshape = x1.reshaped(to: TensorShape(x1.shape.contiguousSize / (7 * 7 * 256), 7, 7, 256))
        let x2 = leakyRelu(x1Reshape.sequenced(through: transConv2D1, flatten, batchNorm2))
        let x2Reshape = x2.reshaped(to: TensorShape(x2.shape.contiguousSize / (7 * 7 * 128), 7, 7, 128))
        let x3 = leakyRelu(x2Reshape.sequenced(through: transConv2D2, flatten, batchNorm3))
        let x3Reshape = x3.reshaped(to: TensorShape(x3.shape.contiguousSize / (14 * 14 * 64), 14, 14, 64))
        return tanh(transConv2D3(x3Reshape))
    }
}

@differentiable
func generatorLoss(fakeLabels: Tensor<Float>) -> Tensor<Float> {
    sigmoidCrossEntropy(logits: fakeLabels,
                        labels: Tensor(ones: fakeLabels.shape))
}

// MARK: Discriminator

struct Discriminator: Layer {
    var conv2D1 = Conv2D<Float>(
        filterShape: (5, 5, 1, 64),
        strides: (2, 2),
        padding: .same
    )
    // leaky relu
    var dropout = Dropout<Float>(probability: 0.3)

    var conv2D2 = Conv2D<Float>(
        filterShape: (5, 5, 64, 128),
        strides: (2, 2),
        padding: .same
    )
    // leaky relu
    // dropout

    var flatten = Flatten<Float>()
    var dense = Dense<Float>(inputSize: 6272, outputSize: 1)

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let x1 = dropout(leakyRelu(conv2D1(input)))
        let x2 = dropout(leakyRelu(conv2D2(x1)))
        return x2.sequenced(through: flatten, dense)
    }
}

@differentiable
func discriminatorLoss(realLabels: Tensor<Float>, fakeLabels: Tensor<Float>) -> Tensor<Float> {
    let realLoss = sigmoidCrossEntropy(logits: realLabels,
                                       labels: Tensor(ones: realLabels.shape)) // should say it's real, 1
    let fakeLoss = sigmoidCrossEntropy(logits: fakeLabels,
                                       labels: Tensor(zeros: fakeLabels.shape)) // should say it's fake, 0
    return realLoss + fakeLoss // accumaltive
}

// MARK: - Training

// create instances of models
var discriminator = Discriminator()
var generator = Generator()

// optimizers
let optG = Adam(for: generator, learningRate: 0.0001)
let optD = Adam(for: discriminator, learningRate: 0.0001)

// test noise so we can track progress
let noise = Tensor<Float>(randomNormal: TensorShape(1, zDim))

let epochs = 20
for epoch in 0 ... epochs {
    Context.local.learningPhase = .training
    for i in 0 ..< (mnist.trainingSize / batchSize) + 1 {
        let realImages = mnist.trainingImages.minibatch(at: i, batchSize: i * batchSize >= mnist.trainingSize ? (mnist.trainingSize - ((i - 1) * batchSize)) : batchSize)

        // train generator
        let noiseG = Tensor<Float>(randomNormal: TensorShape(batchSize, zDim))
        let 𝛁generator = generator.gradient { generator -> Tensor<Float> in
            let fakeImages = generator(noiseG)
            let fakeLabels = discriminator(fakeImages)
            let loss = generatorLoss(fakeLabels: fakeLabels)
            return loss
        }
        optG.update(&generator, along: 𝛁generator)

        // train discriminator
        let noiseD = Tensor<Float>(randomNormal: TensorShape(batchSize, zDim))
        let fakeImages = generator(noiseD)

        let 𝛁discriminator = discriminator.gradient { discriminator -> Tensor<Float> in
            let realLabels = discriminator(realImages)
            let fakeLabels = discriminator(fakeImages)
            let loss = discriminatorLoss(realLabels: realLabels, fakeLabels: fakeLabels)
            return loss
        }
        optD.update(&discriminator, along: 𝛁discriminator)
    }

    // test
    Context.local.learningPhase = .inference

    // render images
    let generatedImage = generator(noise)
    plt.imshow(generatedImage.reshaped(to: TensorShape(28, 28)).makeNumpyArray())
    plt.show()

    // print loss
    let generatorLoss_ = generatorLoss(fakeLabels: generatedImage)
    print("epoch: \(epoch) | Generator loss: \(generatorLoss_)")
}

// Generate another image
let noise1 = Tensor<Float>(randomNormal: TensorShape(1, 100))
let generatedImage = generator(noise1)
plt.imshow(generatedImage.reshaped(to: TensorShape(28, 28)).makeNumpyArray())
plt.savefig("final.png")
