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
import TensorFlow
import Datasets

let options = Options.parseOrExit()

let datasetFolder: URL
let trainFolderA: URL
let trainFolderB: URL
let testFolderA: URL
let testFolderB: URL

if options.datasetPath.length != 0 {
    datasetFolder = URL(fileURLWithPath: options.datasetPath, isDirectory: true)
    trainFolderA = datasetFolder.appendingPathComponent("trainA")
    trainFolderB = datasetFolder.appendingPathComponent("trainB")
    testFolderA = datasetFolder.appendingPathComponent("testA")
    testFolderB = datasetFolder.appendingPathComponent("testB")
} else {
    func downloadZebraDataSetIfNotPresent(to directory: URL) {
        let downloadPath = directory.appendingPathComponent("horse2zebra").path
        let directoryExists = FileManager.default.fileExists(atPath: downloadPath)
        let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: downloadPath)
        let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)

        guard !directoryExists || directoryEmpty else { return }

        let location = URL(
            string: "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip")!
        let _ = DatasetUtilities.downloadResource(
            filename: "horse2zebra", fileExtension: "zip",
            remoteRoot: location.deletingLastPathComponent(), localStorageDirectory: directory)
    }

    datasetFolder = DatasetUtilities.defaultDirectory.appendingPathComponent("CycleGAN", isDirectory: true)
    downloadZebraDataSetIfNotPresent(to: datasetFolder)
    trainFolderA = datasetFolder.appendingPathComponent("horse2zebra/trainA")
    trainFolderB = datasetFolder.appendingPathComponent("horse2zebra/trainB")
    testFolderA = datasetFolder.appendingPathComponent("horse2zebra/testA")
    testFolderB = datasetFolder.appendingPathComponent("horse2zebra/testB")
}

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

var sampleImage = trainDatasetA.batcher.dataset[0].expandingShape(at: 0)
let sampleImageURL = URL(string: FileManager.default.currentDirectoryPath)!.appendingPathComponent("sample.jpg")

// MARK: Train

for epoch in 0 ..< epochs {
    print("Epoch \(epoch) started at: \(Date())")
    Context.local.learningPhase = .training

    let zippedAB = zip(trainDatasetA.batcher.sequenced(), trainDatasetB.batcher.sequenced())
    
    for batch in zippedAB {
        Context.local.learningPhase = .training
        
        let inputX = batch.0
        let inputY = batch.1

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

        // MARK: Inference

        if step % options.sampleLogPeriod == 0 {
            Context.local.learningPhase = .inference
            
            let fakeSample = generatorG(sampleImage) * 0.5 + 0.5

            let fakeSampleImage = Image(tensor: fakeSample[0] * 255)
            fakeSampleImage.save(to: sampleImageURL, format: .rgb)

            print("GeneratorG loss: \(gLoss.scalars[0])")
            print("GeneratorF loss: \(fLoss.scalars[0])")
            print("DiscriminatorX loss: \(xLoss.scalars[0])")
            print("DiscriminatorY loss: \(yLoss.scalars[0])")
        }

        step += 1
    }
}

// MARK: Final test

let testDatasetA = try Images(folderURL: testFolderA).batcher.sequenced()
let testDatasetB = try Images(folderURL: testFolderB).batcher.sequenced()

let zippedTest = zip(testDatasetA, testDatasetB)

let aResultsFolder = try createDirectoryIfNeeded(path: FileManager.default
                                                                  .currentDirectoryPath + "/testA_results")
let bResultsFolder = try createDirectoryIfNeeded(path: FileManager.default
                                                                  .currentDirectoryPath + "/testB_results")

var testStep = 0
for testBatch in zippedTest {
    let realX = testBatch.0
    let realY = testBatch.1

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
