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
import ModelSupport
import TensorFlow

let options = Options.parseOrExit()

let datasetFolder: URL
let trainFolderA: URL
let trainFolderB: URL
let testFolderA: URL
let testFolderB: URL

if let datasetPath = options.datasetPath {
    datasetFolder = URL(fileURLWithPath: datasetPath, isDirectory: true)
    trainFolderA = datasetFolder.appendingPathComponent("trainA")
    testFolderA = datasetFolder.appendingPathComponent("testA")
    trainFolderB = datasetFolder.appendingPathComponent("trainB")
    testFolderB = datasetFolder.appendingPathComponent("testB")
} else {
    func downloadFacadesDataSetIfNotPresent(to directory: URL) {
        let downloadPath = directory.appendingPathComponent("facades").path
        let directoryExists = FileManager.default.fileExists(atPath: downloadPath)
        let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: downloadPath)
        let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)

        guard !directoryExists || directoryEmpty else { return }

        let location = URL(
            string: "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/facades.zip")!
        let _ = DatasetUtilities.downloadResource(
            filename: "facades", fileExtension: "zip",
            remoteRoot: location.deletingLastPathComponent(), localStorageDirectory: directory)
    }

    datasetFolder = DatasetUtilities.defaultDirectory.appendingPathComponent(
        "pix2pix", isDirectory: true)
    downloadFacadesDataSetIfNotPresent(to: datasetFolder)
    trainFolderA = datasetFolder.appendingPathComponent("facades/trainA", isDirectory: true)
    testFolderA = datasetFolder.appendingPathComponent("facades/testA", isDirectory: true)
    trainFolderB = datasetFolder.appendingPathComponent("facades/trainB", isDirectory: true)
    testFolderB = datasetFolder.appendingPathComponent("facades/testB", isDirectory: true)
}

var generator = NetG(inputChannels: 3, outputChannels: 3, ngf: 64, useDropout: false)
var discriminator = NetD(inChannels: 6, lastConvFilters: 64)

let optimizerG = Adam(for: generator, learningRate: 0.0002, beta1: 0.5)
let optimizerD = Adam(for: discriminator, learningRate: 0.0002, beta1: 0.5)

let batchSize = 1
let lambdaL1 = Tensor<Float>(100)

let trainDataset = try PairedImages(folderAURL: trainFolderA, folderBURL: trainFolderB)
let testDataset = try PairedImages(folderAURL: testFolderA, folderBURL: testFolderB)

var sampleImage = testDataset.batcher.dataset[0].source.expandingShape(at: 0)
let sampleImageURL = URL(string: FileManager.default.currentDirectoryPath)!.appendingPathComponent(
    "sample.jpg")

var step = 0

for epoch in 0..<options.epochs {
    print("Epoch \(epoch) started at: \(Date())")

    var discriminatorTotalLoss = Tensor<Float>(0)
    var generatorTotalLoss = Tensor<Float>(0)
    var discriminatorCount = 0

    for batch in trainDataset.batcher.sequenced() {
        defer { step += 1 }

        Context.local.learningPhase = .training

        let concatanatedImages = batch.source.concatenated(with: batch.target)

        let scaledImages = _Raw.resizeNearestNeighbor(images: concatanatedImages, size: [286, 286])
        var croppedImages = scaledImages.slice(
            lowerBounds: Tensor<Int32>([0, Int32(random() % 30), Int32(random() % 30), 0]),
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

            let ganLoss = sigmoidCrossEntropy(
                logits: fakePrediction,
                labels: Tensor<Float>.one.broadcasted(to: fakePrediction.shape))
            let l1Loss =
                meanAbsoluteError(
                    predicted: fakeImages,
                    expected: targetImages) * lambdaL1

            generatorTotalLoss += ganLoss + l1Loss
            return ganLoss + l1Loss
        }

        let fakeImages = generator(sourceImages)
        let descriminatorGradient = TensorFlow.gradient(at: discriminator) { d -> Tensor<Float> in
            let fakeAB = sourceImages.concatenated(
                with: fakeImages,
                alongAxis: 3)
            let fakePrediction = d(fakeAB)
            let fakeLoss = sigmoidCrossEntropy(
                logits: fakePrediction,
                labels: Tensor<Float>.zero.broadcasted(to: fakePrediction.shape))

            let realAB = sourceImages.concatenated(
                with: targetImages,
                alongAxis: 3)
            let realPrediction = d(realAB)
            let realLoss = sigmoidCrossEntropy(
                logits: realPrediction,
                labels: Tensor<Float>.one.broadcasted(to: fakePrediction.shape))

            discriminatorTotalLoss += (fakeLoss + realLoss) * 0.5

            return (fakeLoss + realLoss) * 0.5
        }

        optimizerG.update(&generator, along: generatorGradient)
        optimizerD.update(&discriminator, along: descriminatorGradient)

        // MARK: Sample Inference

        if step % options.sampleLogPeriod == 0 {
            Context.local.learningPhase = .inference

            let fakeSample = generator(sampleImage) * 0.5 + 0.5

            let fakeSampleImage = Image(tensor: fakeSample[0] * 255)
            fakeSampleImage.save(to: sampleImageURL, format: .rgb)
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

let resultsFolder = try createDirectoryIfNeeded(
    path: FileManager.default.currentDirectoryPath + "/results")
for batch in testDataset.batcher.sequenced() {
    let fakeImages = generator(batch.source)

    let tensorImage =
        batch.source
        .concatenated(
            with: fakeImages,
            alongAxis: 2) / 2.0 + 0.5

    let image = Image(tensor: (tensorImage * 255)[0])
    let saveURL = resultsFolder.appendingPathComponent("\(count).jpg", isDirectory: false)
    image.save(to: saveURL, format: .rgb)

    let ganLoss = sigmoidCrossEntropy(
        logits: fakeImages,
        labels: Tensor.one.broadcasted(to: fakeImages.shape))
    let l1Loss =
        meanAbsoluteError(
            predicted: fakeImages,
            expected: batch.target) * lambdaL1

    totalLoss += ganLoss + l1Loss
    count += 1
}

let testLoss = totalLoss / Float(count)
print("Generator test loss: \(testLoss.scalars[0])")
