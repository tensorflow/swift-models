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

// Based on the paper: "Auto-Encoding Variational Bayes"
// by Diederik P Kingma and Max Welling
// Reference implementation: https://github.com/pytorch/examples/blob/master/vae/main.py

import Datasets
import Foundation
import ModelSupport
import TensorFlow

let epochCount = 10
let imageHeight = 28
let imageWidth = 28

let outputFolder = "./output/"
let dataset = MNIST(batchSize: 128, flattening: true)

let inputDim = 784  // 28*28 for any MNIST
let hiddenDim = 400
let latentDim = 20

// Variational Autoencoder
public struct VAE: Layer {
    // Encoder
    public var encoderDense1: Dense<Float>
    public var encoderDense2_1: Dense<Float>
    public var encoderDense2_2: Dense<Float>
    // Decoder
    public var decoderDense1: Dense<Float>
    public var decoderDense2: Dense<Float>

    public init() {
        self.encoderDense1 = Dense<Float>(
            inputSize: inputDim, outputSize: hiddenDim, activation: relu)
        self.encoderDense2_1 = Dense<Float>(inputSize: hiddenDim, outputSize: latentDim)
        self.encoderDense2_2 = Dense<Float>(inputSize: hiddenDim, outputSize: latentDim)

        self.decoderDense1 = Dense<Float>(
            inputSize: latentDim, outputSize: hiddenDim, activation: relu)
        self.decoderDense2 = Dense<Float>(inputSize: hiddenDim, outputSize: inputDim)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> [Tensor<Float>] {
        // Encode
        let intermediateInput = encoderDense1(input)
        let mu = encoderDense2_1(intermediateInput)
        let logVar = encoderDense2_2(intermediateInput)

        // Re-parameterization trick
        let std = exp(0.5 * logVar)
        let epsilon = Tensor<Float>(randomNormal: std.shape)
        let z = mu + epsilon * std

        // Decode
        let output = z.sequenced(through: decoderDense1, decoderDense2)
        return [output, mu, logVar]
    }
}

var vae = VAE()
let optimizer = Adam(for: vae, learningRate: 1e-3)

// Loss function: sum of the KL divergence of the embeddings and the cross entropy loss between the input and it's reconstruction. 
func vaeLossFunction(
    input: Tensor<Float>, output: Tensor<Float>, mu: Tensor<Float>, logVar: Tensor<Float>
) -> Tensor<Float> {
    let crossEntropy = sigmoidCrossEntropy(logits: output, labels: input, reduction: _sum)
    let klDivergence = -0.5 * (1 + logVar - pow(mu, 2) - exp(logVar)).sum()
    return crossEntropy + klDivergence
}

let individualTestImages = dataset.testDataset.batched(1)
var testImageIterator = individualTestImages.makeIterator()

// Training loop
for epoch in 1...epochCount {
    // Test for each epoch
    if let nextIndividualImage = testImageIterator.next() {
        let sampleTensor = nextIndividualImage.data
        let sampleImage = Tensor(
            shape: [1, imageHeight * imageWidth], scalars: sampleTensor.scalars)

        let testOutputs = vae(sampleImage)
        let testImage = testOutputs[0]
        let testMu = testOutputs[1]
        let testLogVar = testOutputs[2]
        if epoch == 1 || epoch % 10 == 0 {
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
        }

        let sampleLoss = vaeLossFunction(
            input: sampleImage, output: testImage, mu: testMu, logVar: testLogVar)
        print("[Epoch: \(epoch)] Loss: \(sampleLoss)")
    }

    for batch in dataset.training.sequenced() {
        let x = batch.first

        let 𝛁model = TensorFlow.gradient(at: vae) { vae -> Tensor<Float> in
            let outputs = vae(x)
            let output = outputs[0]
            let mu = outputs[1]
            let logVar = outputs[2]
            return vaeLossFunction(input: x, output: output, mu: mu, logVar: logVar)
        }

        optimizer.update(&vae, along: 𝛁model)
    }
}
