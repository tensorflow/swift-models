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
import TensorFlow

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

public struct CIFAR10 {
    public let trainingDataset: Dataset<CIFARExample>
    public let testDataset: Dataset<CIFARExample>

    public init() {
        self.trainingDataset = Dataset<CIFARExample>(elements: loadCIFARTrainingFiles())
        self.testDataset = Dataset<CIFARExample>(elements: loadCIFARTestFile())
    }
}

func downloadCIFAR10IfNotPresent(to directory: String = ".") {
    let downloadPath = "\(directory)/cifar-10-batches-bin"
    let directoryExists = FileManager.default.fileExists(atPath: downloadPath)

    guard !directoryExists else { return }

    print("Downloading CIFAR dataset...")
    let archivePath = "\(directory)/cifar-10-binary.tar.gz"
    let archiveExists = FileManager.default.fileExists(atPath: archivePath)
    if !archiveExists {
        print("Archive missing, downloading...")
        do {
            let downloadedFile = try Data(
                contentsOf: URL(
                    string: "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz")!)
            try downloadedFile.write(to: URL(fileURLWithPath: archivePath))
        } catch {
            print("Could not download CIFAR dataset, error: \(error)")
            exit(-1)
        }
    }

    print("Archive downloaded, processing...")

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
        print("CIFAR extraction failed with error: \(error)")
    }

    do {
        try FileManager.default.removeItem(atPath: archivePath)
    } catch {
        print("Could not remove archive, error: \(error)")
        exit(-1)
    }

    print("Unarchiving completed")
}

func loadCIFARFile(named name: String, in directory: String = ".") -> CIFARExample {
    downloadCIFAR10IfNotPresent(to: directory)
    let path = "\(directory)/cifar-10-batches-bin/\(name)"

    let imageCount = 10000
    guard let fileContents = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
        print("Could not read dataset file: \(name)")
        exit(-1)
    }
    guard fileContents.count == 30_730_000 else {
        print(
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

    let mean = Tensor<Float>([0.485, 0.456, 0.406])
    let std = Tensor<Float>([0.229, 0.224, 0.225])
    let imagesNormalized = ((imageTensor / 255.0) - mean) / std

    return CIFARExample(label: Tensor<Int32>(labelTensor), data: imagesNormalized)
}

func loadCIFARTrainingFiles() -> CIFARExample {
    let data = (1..<6).map { loadCIFARFile(named: "data_batch_\($0).bin") }
    return CIFARExample(
        label: Tensor(concatenating: data.map { $0.label }, alongAxis: 0),
        data: Tensor(concatenating: data.map { $0.data }, alongAxis: 0)
    )
}

func loadCIFARTestFile() -> CIFARExample {
    return loadCIFARFile(named: "test_batch.bin")
}
