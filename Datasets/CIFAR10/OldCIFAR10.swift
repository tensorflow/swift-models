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
import Batcher

public struct OldCIFAR10: ImageClassificationDataset {
    public typealias SourceDataSet = [TensorPair<Float, Int32>]
    public let training: Batcher<SourceDataSet>
    public let test: Batcher<SourceDataSet>

    public init(batchSize: Int) {
        self.init(
            batchSize: batchSize,
            remoteBinaryArchiveLocation: URL(
                string: "https://storage.googleapis.com/s4tf-hosted-binaries/datasets/CIFAR10/cifar-10-binary.tar.gz")!, 
            normalizing: true)
    }

    public init(
        batchSize: Int,
        remoteBinaryArchiveLocation: URL, 
        localStorageDirectory: URL = DatasetUtilities.defaultDirectory
                .appendingPathComponent("CIFAR10", isDirectory: true), 
        normalizing: Bool) 
    {
        _downloadCIFAR10IfNotPresent(from: remoteBinaryArchiveLocation, to: localStorageDirectory)
        self.training = Batcher(
            on: _loadCIFARTrainingFiles(localStorageDirectory: localStorageDirectory, normalizing: normalizing),
            batchSize: batchSize,
            numWorkers: 1, //No need to use parallelism since everything is loaded in memory
            shuffle: true)
        self.test = Batcher(
            on: _loadCIFARTestFile(localStorageDirectory: localStorageDirectory, normalizing: normalizing),
            batchSize: batchSize,
            numWorkers: 1) //No need to use parallelism since everything is loaded in memory
    }
}

func _downloadCIFAR10IfNotPresent(from location: URL, to directory: URL) {
    let downloadPath = directory.appendingPathComponent("cifar-10-batches-bin").path
    let directoryExists = FileManager.default.fileExists(atPath: downloadPath)
    let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: downloadPath)
    let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)

    guard !directoryExists || directoryEmpty else { return }

    let _ = DatasetUtilities.downloadResource(
        filename: "cifar-10-binary", fileExtension: "tar.gz",
        remoteRoot: location.deletingLastPathComponent(), localStorageDirectory: directory)
}

func _loadCIFARFile(named name: String, in directory: URL, normalizing: Bool = true) -> [TensorPair<Float, Int32>] {
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
    var imageTensor = Tensor<Float>(images.transposed(permutation: [0, 2, 3, 1]))

    // The value of mean and std were calculated with the following Swift code:
    // ```
    // import TensorFlow
    // import Datasets
    // import Foundation
    // let urlString = "https://storage.googleapis.com/s4tf-hosted-binaries/datasets/CIFAR10/cifar-10-binary.tar.gz"
    // let cifar = CIFAR10(batchSize: 50000,
    //                     remoteBinaryArchiveLocation: URL(string: urlString)!,
    //                     normalizing: false)
    // for batch in cifar.training.sequenced() {
    //     let images = Tensor<Double>(batch.first) / 255.0
    //     let mom = images.moments(squeezingAxes: [0,1,2])
    //     print("mean: \(mom.mean) std: \(sqrt(mom.variance))")
    // }
    // ```
    if normalizing {
        let mean = Tensor<Float>(
                [0.4913996898,
                 0.4821584196,
                 0.4465309242])
        let std = Tensor<Float>(
                [0.2470322324,
                 0.2434851280,
                 0.2615878417])
        imageTensor = ((imageTensor / 255.0) - mean) / std
    }
    
    return (0..<imageCount).map { TensorPair(first: imageTensor[$0], second: Tensor<Int32>(labelTensor[$0])) }
        
}

func _loadCIFARTrainingFiles(localStorageDirectory: URL, normalizing: Bool = true) -> [TensorPair<Float, Int32>] {
    let data = (1..<6).map {
        _loadCIFARFile(named: "data_batch_\($0).bin", in: localStorageDirectory, normalizing: normalizing)
    }
    return data.reduce([], +)
}

func _loadCIFARTestFile(localStorageDirectory: URL, normalizing: Bool = true) -> [TensorPair<Float, Int32>] {
    return _loadCIFARFile(named: "test_batch.bin", in: localStorageDirectory, normalizing: normalizing)
}