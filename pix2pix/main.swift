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
import pix2pix
import TensorFlow

let options = Options.parseOrExit()

let dataset = try! Pix2PixDataset(
    from: options.datasetPath,
    trainBatchSize: 1,
    testBatchSize: 1)

var validationImage = dataset.testSamples[0].source.expandingShape(at: 0)

var generator = NetG(inputChannels: 3, outputChannels: 3, ngf: 64, useDropout: false)
var discriminator = NetD(inChannels: 6, lastConvFilters: 64)

let optimizerG = Adam(for: generator, learningRate: 0.0002, beta1: 0.5)
let optimizerD = Adam(for: discriminator, learningRate: 0.0002, beta1: 0.5)

let epochCount = options.epochs
var step = 0
let lambdaL1 = Tensor<Float>(100)
fileprivate let writeCheckPoint = true

for (epoch, epochBatches) in dataset.training.prefix(epochCount).enumerated() {
    print("Epoch \(epoch) started at: \(Date())")
    
    var discriminatorTotalLoss = Tensor<Float>(0)
    var generatorTotalLoss = Tensor<Float>(0)
    var discriminatorCount = 0
    
    for batch in epochBatches {
        print("Batch \(step) started at \(Date())")
        defer { step += 1 }
        Context.local.learningPhase = .training
        let concatanatedImages = batch.source.concatenated(with: batch.target)
        let scaledImages = _Raw.resizeNearestNeighbor(images: concatanatedImages, size: [286, 286])
        var croppedImages = scaledImages.slice(lowerBounds: Tensor<Int32>([0, Int32.random(in: 0...29), Int32.random(in: 0...29), 0]),
                                               sizes: [2, 256, 256, 3])
        if Bool.random() {
            croppedImages = _Raw.reverse(croppedImages, dims: [false, false, true, false])
        }
        
        let sourceImages = croppedImages[0].expandingShape(at: 0)
        let targetImages = croppedImages[1].expandingShape(at: 0)
        
        let generatorGradient = TensorFlow.gradient(at: generator) { g -> Tensor<Float> in
            let fakeImages = g(sourceImages)
            let fakeAB = sourceImages.concatenated(with: fakeImages, alongAxis: 3)
            let fakePrediction = discriminator(fakeAB)
            
            let ganLoss = sigmoidCrossEntropy(logits: fakePrediction,
                                              labels: Tensor<Float>.one.broadcasted(to: fakePrediction.shape))
            let l1Loss = meanAbsoluteError(predicted: fakeImages,
                                           expected: targetImages) * lambdaL1
            
            generatorTotalLoss += ganLoss + l1Loss
            return ganLoss + l1Loss
        }
        
        let fakeImages = generator(sourceImages)
                
        let descriminatorGradient = TensorFlow.gradient(at: discriminator) { d -> Tensor<Float> in
            let fakeAB = sourceImages.concatenated(with: fakeImages,
                                                   alongAxis: 3)
            let fakePrediction = d(fakeAB)
            let fakeLoss = sigmoidCrossEntropy(logits: fakePrediction,
                                               labels: Tensor<Float>.zero.broadcasted(to: fakePrediction.shape))
            let realAB = sourceImages.concatenated(with: targetImages,
                                                   alongAxis: 3)
            let realPrediction = d(realAB)
            let realLoss = sigmoidCrossEntropy(logits: realPrediction,
                                               labels: Tensor<Float>.one.broadcasted(to: fakePrediction.shape))
            discriminatorTotalLoss += (fakeLoss + realLoss) * 0.5
            return (fakeLoss + realLoss) * 0.5
        }
        
        optimizerG.update(&generator, along: generatorGradient)
        optimizerD.update(&discriminator, along: descriminatorGradient)
                
        // MARK: Sample Inference
        if step % options.sampleLogPeriod == 0 {
            Context.local.learningPhase = .inference
            let fakeSample = generator(validationImage) * 0.5 + 0.5
            try fakeSample[0].scaled(by: 255).saveImage(directory: "output", name: "sample" + String(epoch) + String(step))
        }
        discriminatorCount += 1
    }
    
    let generatorLoss = generatorTotalLoss / Float(discriminatorCount)
    let discriminatorLoss = discriminatorTotalLoss / Float(discriminatorCount)
    print("Generator train loss: \(generatorLoss.scalars[0])")
    print("Discriminator train loss: \(discriminatorLoss.scalars[0])")
}

Context.local.learningPhase = .inference

var totalLoss = Tensor<Float>(0)
var count = 0

for batch in dataset.testing {
    let fakeImages = generator(batch.source)

    let tensorImage = batch.source
                           .concatenated(with: fakeImages,
                                         alongAxis: 2) / 2.0 + 0.5

    try tensorImage[0].scaled(by: 255).saveImage(directory: "output/results", name: "\(count)")

    let ganLoss = sigmoidCrossEntropy(logits: fakeImages,
                                      labels: Tensor.one.broadcasted(to: fakeImages.shape))
    let l1Loss = meanAbsoluteError(predicted: fakeImages,
                                   expected: batch.target) * lambdaL1

    totalLoss += ganLoss + l1Loss
    count += 1
}

let testLoss = totalLoss / Float(count)
print("Generator test loss: \(testLoss.scalars[0])")

// MARK: Checkpoint
if writeCheckPoint {
    do {
        let temporaryDirectory = FileManager.default.temporaryDirectory.appendingPathComponent("NetG")
        try generator.writeCheckpoint(to: temporaryDirectory, name: "NetG")
    } catch {
        fatalError("ERROR: checkpoint failed")
    }
}

