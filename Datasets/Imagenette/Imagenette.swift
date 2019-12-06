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

// Original source:
// "Imagenette"
// Jeremy Howard
// https://github.com/fastai/imagenette

import Foundation
import ModelSupport
import TensorFlow

public struct Imagenette: ImageClassificationDataset {
    public let trainingDataset: Dataset<LabeledExample>
    public let testDataset: Dataset<LabeledExample>
    public let trainingExampleCount = 12894
    public let validationExampleCount = 500

    public enum ImageSize {
        case full
        case resized160
        case resized320

        var suffix: String {
            switch self {
            case .full: return ""
            case .resized160: return "-160"
            case .resized320: return "-320"
            }
        }
    }

    public init() {
        self.init(inputSize: .resized320, outputSize: 224)
    }

    public init(inputSize: ImageSize, outputSize: Int) {
        do {
            self.trainingDataset = Dataset<LabeledExample>(
                elements: try loadImagenetteTrainingImages(
                    inputSize: inputSize, outputSize: outputSize))
            self.testDataset = Dataset<LabeledExample>(
                elements: try loadImagenetteValidationImages(
                    inputSize: inputSize, outputSize: outputSize))
        } catch {
            fatalError("Could not load Imagenette dataset: \(error)")
        }
    }
}

func downloadImagenetteIfNotPresent(to directory: String = ".", size: Imagenette.ImageSize) {
    let downloadPath = "\(directory)/imagenette\(size.suffix)"
    let directoryExists = FileManager.default.fileExists(atPath: downloadPath)

    guard !directoryExists else { return }

    printError("Downloading Imagenette dataset...")
    let archivePath = "\(directory)/imagenette\(size.suffix).tgz"
    let archiveExists = FileManager.default.fileExists(atPath: archivePath)
    if !archiveExists {
        printError("Archive missing, downloading...")
        do {
            let downloadedFile = try Data(
                contentsOf: URL(
                    string:
                        "https://s3.amazonaws.com/fast-ai-imageclas/imagenette\(size.suffix).tgz")!)
            try downloadedFile.write(to: URL(fileURLWithPath: archivePath))
        } catch {
            printError("Could not download Imagenette dataset, error: \(error)")
            exit(-1)
        }
    }

    printError("Archive downloaded, processing...")

    #if os(macOS)
        let tarLocation = "/usr/bin/tar"
    #else
        let tarLocation = "/bin/tar"
    #endif

    let task = Process()
    task.executableURL = URL(fileURLWithPath: tarLocation)
    task.arguments = ["xzf", archivePath]
    do {
        try task.run()
        task.waitUntilExit()
    } catch {
        printError("Imagenette extraction failed with error: \(error)")
    }

    do {
        try FileManager.default.removeItem(atPath: archivePath)
    } catch {
        printError("Could not remove archive, error: \(error)")
        exit(-1)
    }

    printError("Unarchiving completed")
}

func loadImagenetteDirectory(
    named name: String, in directory: String = ".", inputSize: Imagenette.ImageSize, outputSize: Int
) throws -> LabeledExample {
    downloadImagenetteIfNotPresent(to: directory, size: inputSize)
    let path = URL(fileURLWithPath: "\(directory)/imagenette\(inputSize.suffix)/\(name)")

    let dirContents = try FileManager.default.contentsOfDirectory(
        at: path, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles])

    var imageData: [Float] = []
    var stringLabels: [String] = []
    var labels: [Int32] = []
    var currentLabel: Int32 = 0
    var imageCount = 0
    for directoryURL in dirContents {
        stringLabels.append(directoryURL.lastPathComponent)

        let subdirContents = try FileManager.default.contentsOfDirectory(
            at: directoryURL, includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles])
        for fileURL in subdirContents {
            let image = Image(jpeg: fileURL)
            let resizedImage = image.resized(to: (outputSize, outputSize))
            let scaledImage = resizedImage.tensor / 255.0
            imageData.append(contentsOf: scaledImage.scalars)

            labels.append(currentLabel)

            imageCount += 1
        }
        currentLabel += 1
    }

    let labelTensor = Tensor<Int32>(shape: [imageCount], scalars: labels)
    let imageTensor = Tensor<Float>(
        shape: [imageCount, outputSize, outputSize, 3], scalars: imageData)

    return LabeledExample(label: labelTensor, data: imageTensor)
}

func loadImagenetteTrainingImages(inputSize: Imagenette.ImageSize, outputSize: Int) throws
    -> LabeledExample
{
    return try loadImagenetteDirectory(named: "train", inputSize: inputSize, outputSize: outputSize)
}

func loadImagenetteValidationImages(inputSize: Imagenette.ImageSize, outputSize: Int) throws
    -> LabeledExample
{
    return try loadImagenetteDirectory(named: "val", inputSize: inputSize, outputSize: outputSize)
}
