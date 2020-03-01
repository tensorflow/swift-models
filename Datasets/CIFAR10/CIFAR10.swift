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
// "The CIFAR-10 dataset"
// Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.
// https://www.cs.toronto.edu/~kriz/cifar.html

import Foundation
import ModelSupport
import TensorFlow

public struct CIFAR10: ImageClassificationDataset {
    public let trainingDataset: Dataset<LabeledExample>
    public let testDataset: Dataset<LabeledExample>
    public let trainingExampleCount = 50000
    public let testExampleCount = 10000

    public init() {
        self.init(
            normalizing: true, 
            remoteBinaryArchiveLocation: URL(
                string: "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz")!)
    }

    public init(remoteBinaryArchiveLocation: URL) {
        self.init(
            remoteBinaryArchiveLocation: remoteBinaryArchiveLocation,
            localStorageDirectory: FileManager.default.temporaryDirectory.appendingPathComponent(
                "CIFAR10", isDirectory: true))
    }

    public init(remoteBinaryArchiveLocation: URL, localStorageDirectory: URL, normalizing: Bool) {
        downloadCIFAR10IfNotPresent(from: remoteBinaryArchiveLocation, to: localStorageDirectory)
        self.trainingDataset = Dataset<LabeledExample>(
            elements: loadCIFARTrainingFiles(localStorageDirectory: localStorageDirectory, normalizing: normalizing))
        self.testDataset = Dataset<LabeledExample>(
            elements: loadCIFARTestFile(localStorageDirectory: localStorageDirectory, normalizing: normalizing))
    }
}

func downloadCIFAR10IfNotPresent(from location: URL, to directory: URL) {
    let downloadPath = directory.appendingPathComponent("cifar-10-batches-bin").path
    let directoryExists = FileManager.default.fileExists(atPath: downloadPath)
    let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: downloadPath)
    let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)

    guard !directoryExists || directoryEmpty else { return }

    let _ = DatasetUtilities.downloadResource(
        filename: "cifar-10-binary", fileExtension: "tar.gz",
        remoteRoot: location.deletingLastPathComponent(), localStorageDirectory: directory)
}

func loadCIFARFile(named name: String, in directory: URL, normalizing: Bool = true) -> LabeledExample {
    let path = directory.appendingPathComponent("cifar-10-batches-bin/\(name)").path

    let imageCount = 10000
    guard let fileContents = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
        printError("Could not read dataset file: \(name)")
        exit(-1)
    }
    guard fileContents.count == 30_730_000 else {
        printError(
            "Dataset file \(name) should have 30730000 bytes, instead had \(fileContents.count)")
        exit(-1)
    }

    var bytes: [UInt8] = []
    var labels: [Int64] = []

    let imageByteSize = 3073
    for imageIndex in 0..<imageCount {
        let baseAddress = imageIndex * imageByteSize
        labels.append(Int64(fileContents[baseAddress]))
        bytes.append(contentsOf: fileContents[(baseAddress + 1)..<(baseAddress + 3073)])
    }

    let labelTensor = Tensor<Int64>(shape: [imageCount], scalars: labels)
    let images = Tensor<UInt8>(shape: [imageCount, 3, 32, 32], scalars: bytes)

    // Transpose from the CIFAR-provided N(CHW) to TF's default NHWC.
    let imageTensor = Tensor<Float>(images.transposed(permutation: [0, 2, 3, 1]))

    if normalizing {
        let mean = Tensor<Float>([0.485, 0.456, 0.406])
        let std = Tensor<Float>([0.229, 0.224, 0.225])
        let imagesNormalized = ((imageTensor / 255.0) - mean) / std
        return LabeledExample(label: Tensor<Int32>(labelTensor), data: imagesNormalized)
    }
    else {
        return LabeledExample(label: Tensor<Int32>(labelTensor), data: imageTensor)
    }
        
}

func loadCIFARTrainingFiles(localStorageDirectory: URL, normalizing: Bool = true) -> LabeledExample {
    let data = (1..<6).map {
        loadCIFARFile(named: "data_batch_\($0).bin", in: localStorageDirectory, normalizing: normalizing)
    }
    return LabeledExample(
        label: Tensor(concatenating: data.map { $0.label }, alongAxis: 0),
        data: Tensor(concatenating: data.map { $0.data }, alongAxis: 0)
    )
}

func loadCIFARTestFile(localStorageDirectory: URL, normalizing: Bool = true) -> LabeledExample {
    return loadCIFARFile(named: "test_batch.bin", in: localStorageDirectory, normalizing: normalizing)
}
