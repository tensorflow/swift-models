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
import ModelSupport
import TensorBoardX
import TensorFlow

let options = Options.parseOrExit()
let logDirURL = URL(fileURLWithPath: options.tensorboardLogdir, isDirectory: true)
let runId = currentRunId(logDir: logDirURL)
let writerURL = logDirURL.appendingPathComponent(String(runId), isDirectory: true)
let writer = SummaryWriter(logdir: writerURL)

print("Starting with run id: \(runId)")

let datasetFolder = URL(fileURLWithPath: options.datasetPath, isDirectory: true)
let trainFolderA = datasetFolder.appendingPathComponent("trainA")
let trainFolderB = datasetFolder.appendingPathComponent("trainB")
let testFolderA = datasetFolder.appendingPathComponent("testA")
let testFolderB = datasetFolder.appendingPathComponent("testB")

let trainDatasetA = try Images(folderURL: trainFolderA)
let trainDatasetB = try Images(folderURL: trainFolderB)

var generatorG = ResNetGenerator(inputChannels: 3, outputChannels: 3, blocks: 9, ngf: 64, normalization: InstanceNorm2D.self)
var generatorF = ResNetGenerator(inputChannels: 3, outputChannels: 3, blocks: 9, ngf: 64, normalization: InstanceNorm2D.self)
var discriminatorX = NetD(inChannels: 3, lastConvFilters: 64)
var discriminatorY = NetD(inChannels: 3, lastConvFilters: 64)

let optimizerGF = Adam(for: generatorF, learningRate: 0.0002, beta1: 0.5)
let optimizerGG = Adam(for: generatorG, learningRate: 0.0002, beta1: 0.5)
let optimizerDX = Adam(for: discriminatorX, learningRate: 0.0002, beta1: 0.5)
let optimizerDY = Adam(for: discriminatorY, learningRate: 0.0002, beta1: 0.5)

let epochs = options.epochs
let batchSize = 1
let lambdaL1 = Tensorf(10)
let _zeros = Tensorf.zero
let _ones = Tensorf.one

var step = 0

var sampleImage: Tensorf = .zero
for sampleBatch in trainDatasetA.dataset.batched(1) {
    sampleImage = sampleBatch.image
    break
}

// MARK: Train

for epoch in 0 ..< epochs {
    print("Epoch \(epoch) started at: \(Date())")
    Context.local.learningPhase = .training

    let trainingAShuffled = trainDatasetA.dataset
        .shuffled(sampleCount: trainDatasetA.count,
                  randomSeed: Int64(epoch))
    let trainingBShuffled = trainDatasetB.dataset
        .shuffled(sampleCount: trainDatasetB.count,
                  randomSeed: Int64(epoch))
    let zippedAB = zip(trainingAShuffled, trainingBShuffled)

    for batch in zippedAB.batched(batchSize) {
        Context.local.learningPhase = .training

        let inputX = batch.first.image
        let inputY = batch.second.image

        // we do it outside of GPU scope so that dataset shuffling happens on CPU side
        let concatanatedImages = inputX.concatenated(with: inputY)

        let scaledImages = resize(images: concatanatedImages,
                                  size: (286, 286),
                                  method: .nearest)
        var croppedImages = scaledImages.slice(lowerBounds: Tensor<Int32>([0, Int32(random() % 30), Int32(random() % 30), 0]),
                                               sizes: [2, 256, 256, 3])
        if Bool.random() {
            croppedImages = croppedImages.reversed(inAxes: 2)
        }

        let realX = croppedImages[0].expandingShape(at: 0)
        let realY = croppedImages[1].expandingShape(at: 0)

        let onesd = _ones.broadcasted(to: [1, 30, 30, 1])
        let zerosd = _zeros.broadcasted(to: [1, 30, 30, 1])

        var _fakeX = Tensorf.zero
        var _fakeY = Tensorf.zero

        let (gLoss, 𝛁generatorG) = valueWithGradient(at: generatorG) { g -> Tensorf in
            let fakeY = g(realX)
            let cycledX = generatorF(fakeY)
            let fakeX = generatorF(realY)
            let cycledY = g(fakeX)

            let cycleConsistencyLoss = (abs(realX - cycledX).mean() +
                abs(realY - cycledY).mean()) * lambdaL1

            let discFakeY = discriminatorY(fakeY)
            let generatorLoss = sigmoidCrossEntropy(logits: discFakeY, labels: onesd)

            let sameY = g(realY)
            let identityLoss = abs(sameY - realY).mean() * lambdaL1 * 0.5

            let totalLoss = cycleConsistencyLoss + generatorLoss + identityLoss

            _fakeX = fakeX

            return totalLoss
        }

        let (fLoss, 𝛁generatorF) = valueWithGradient(at: generatorF) { g -> Tensorf in
            let fakeX = g(realY)
            let cycledY = generatorG(fakeX)
            let fakeY = generatorG(realX)
            let cycledX = g(fakeY)

            let cycleConsistencyLoss = (abs(realY - cycledY).mean()
                + abs(realX - cycledX).mean()) * lambdaL1

            let discFakeX = discriminatorX(fakeX)
            let generatorLoss = sigmoidCrossEntropy(logits: discFakeX, labels: onesd)

            let sameX = g(realX)
            let identityLoss = abs(sameX - realX).mean() * lambdaL1 * 0.5

            let totalLoss = cycleConsistencyLoss + generatorLoss + identityLoss

            _fakeY = fakeY
            return totalLoss
        }

        let (xLoss, 𝛁discriminatorX) = valueWithGradient(at: discriminatorX) { d -> Tensorf in
            let discFakeX = d(_fakeX)
            let discRealX = d(realX)

            let totalLoss = 0.5 * (sigmoidCrossEntropy(logits: discFakeX, labels: zerosd)
                + sigmoidCrossEntropy(logits: discRealX, labels: onesd))

            return totalLoss
        }

        let (yLoss, 𝛁discriminatorY) = valueWithGradient(at: discriminatorY) { d -> Tensorf in
            let discFakeY = d(_fakeY)
            let discRealY = d(realY)

            let totalLoss = 0.5 * (sigmoidCrossEntropy(logits: discFakeY, labels: zerosd)
                + sigmoidCrossEntropy(logits: discRealY, labels: onesd))

            return totalLoss
        }

        optimizerGG.update(&generatorG, along: 𝛁generatorG)
        optimizerGF.update(&generatorF, along: 𝛁generatorF)
        optimizerDX.update(&discriminatorX, along: 𝛁discriminatorX)
        optimizerDY.update(&discriminatorY, along: 𝛁discriminatorY)

        writer.addScalars(mainTag: "train_loss",
                          taggedScalars: [
                              "GeneratorG": gLoss.scalars[0],
                              "GeneratorF": fLoss.scalars[0],
                              "DiscriminatorX": xLoss.scalars[0],
                              "DiscriminatorY": yLoss.scalars[0],
                          ],
                          globalStep: step)

        // MARK: Inference

        if step % options.sampleLogPeriod == 0 {
            let fakeSample = generatorG(sampleImage) * 0.5 + 0.5

            writer.addImages(tag: "reals", images: sampleImage * 0.5 + 0.5, globalStep: step)
            writer.addImages(tag: "fakes", images: fakeSample, globalStep: step)
            writer.flush()
        }

        step += 1
    }
}

// MARK: Final test

let testDatasetA = try Images(folderURL: testFolderA).dataset
let testDatasetB = try Images(folderURL: testFolderB).dataset

let zippedTest = zip(testDatasetA, testDatasetB)

let aResultsFolder = try createDirectoryIfNeeded(path: FileManager.default
                                                                  .currentDirectoryPath + "/testA_results")
let bResultsFolder = try createDirectoryIfNeeded(path: FileManager.default
                                                                  .currentDirectoryPath + "/testB_results")

var testStep = 0
for testBatch in zippedTest.batched(1) {
    let realX = testBatch.first.image
    let realY = testBatch.second.image

    let fakeY = generatorG(realX)
    let fakeX = generatorF(realY)

    let resultX = realX.concatenated(with: fakeY, alongAxis: 2) * 0.5 + 0.5
    let resultY = realY.concatenated(with: fakeX, alongAxis: 2) * 0.5 + 0.5

    let imageX = Image(tensor: resultX[0] * 255)
    let imageY = Image(tensor: resultY[0] * 255)

    imageX.save(to: aResultsFolder.appendingPathComponent("\(String(testStep)).jpg", isDirectory: false),
                format: .rgb)
    imageY.save(to: bResultsFolder.appendingPathComponent("\(String(testStep)).jpg", isDirectory: false),
                format: .rgb)

    testStep += 1
}
