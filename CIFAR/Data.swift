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
import FoundationNetworking
import TensorFlow

func downloadCIFAR10IfNotPresent(to directory: String = ".") {
    let downloadPath = "\(directory)/cifar-10-batches-bin"
    let directoryExists = FileManager.default.fileExists(atPath: downloadPath)

    if !directoryExists {
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
                fatalError("Could not download CIFAR dataset, error: \(error)")
            }
        }

        print("Archive downloaded, processing...")

        #if os(macOS)
            let tarLocation = "/usr/bin/tar"
        #else
            let tarLocation = "/bin/tar"
        #endif

        if #available(macOS 10.13, *) {
            let task = Process()
            task.executableURL = URL(fileURLWithPath: tarLocation)
            task.arguments = ["xzf", archivePath]
            do {
                try task.run()
                task.waitUntilExit()
            } catch {
                print("CIFAR extraction failed with error: \(error)")
            }
        } else {
            fatalError("Process() is missing from this platform")
        }

        do {
            try FileManager.default.removeItem(atPath: archivePath)
        } catch {
            fatalError("Could not remove archive, error: \(error)")
        }

        print("Unarchiving completed")
    }
}

extension Tensor where Scalar: _TensorFlowDataTypeCompatible {
    public var _tfeTensorHandle: _AnyTensorHandle {
        TFETensorHandle(_owning: handle._cTensorHandle)
    }
}

struct Example: TensorGroup {
    var label: Tensor<Int32>
    var data: Tensor<Float>

    init(label: Tensor<Int32>, data: Tensor<Float>) {
        self.label = label
        self.data = data
    }

    public init<C: RandomAccessCollection>(
        _handles: C
    ) where C.Element: _AnyTensorHandle {
        precondition(_handles.count == 2)
        let labelIndex = _handles.startIndex
        let dataIndex = _handles.index(labelIndex, offsetBy: 1)
        label = Tensor<Int32>(handle: TensorHandle<Int32>(handle: _handles[labelIndex]))
        data = Tensor<Float>(handle: TensorHandle<Float>(handle: _handles[dataIndex]))
    }

    public var _tensorHandles: [_AnyTensorHandle] {
        [label._tfeTensorHandle, data._tfeTensorHandle]
    }
}

func loadCIFARFile(named name: String, in directory: String = ".") -> Example {
    downloadCIFAR10IfNotPresent(to: directory)
    let path = "\(directory)/cifar-10-batches-bin/\(name)"

    let imageCount = 10000
    guard let fileContents = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
        fatalError("Could not read dataset file: \(name)")
    }
    guard fileContents.count == 30_730_000 else {
        fatalError(
            "Dataset file \(name) should have 30730000 bytes, instead had \(fileContents.count)")
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

    // transpose from the provided N(CHW) to TF default NHWC
    let imageTensor = Tensor<Float>(
        images.transposed(withPermutations: [0, 2, 3, 1]))

    let mean = Tensor<Float>([0.485, 0.456, 0.406])
    let std = Tensor<Float>([0.229, 0.224, 0.225])
    let imagesNormalized = ((imageTensor / 255.0) - mean) / std

    return Example(label: Tensor<Int32>(labelTensor), data: imagesNormalized)
}

func loadCIFARTrainingFiles() -> Example {
    let data = (1..<6).map { loadCIFARFile(named: "data_batch_\($0).bin") }
    return Example(
        label: Raw.concat(concatDim: Tensor<Int32>(0), data.map { $0.label }),
        data: Raw.concat(concatDim: Tensor<Int32>(0), data.map { $0.data })
    )
}

func loadCIFARTestFile() -> Example {
    return loadCIFARFile(named: "test_batch.bin")
}

func loadCIFAR10() -> (
    training: Dataset<Example>, test: Dataset<Example>
) {
    let trainingDataset = Dataset<Example>(elements: loadCIFARTrainingFiles())
    let testDataset = Dataset<Example>(elements: loadCIFARTestFile())
    return (training: trainingDataset, test: testDataset)
}
