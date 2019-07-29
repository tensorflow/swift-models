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
import TensorFlow

public struct MNIST {
    public let trainImages: Tensor<Float>
    public let trainLabels: Tensor<Int32>
    public let testImages: Tensor<Float>
    public let testLabels: Tensor<Int32>

    public let trainingSize: Int
    public let testSize: Int

    let batchSize: Int

    public init(batchSize: Int, flattening: Bool = false, normalizing: Bool = false) {
        self.batchSize = batchSize

        let (trainImages, trainLabels) = readMNIST(
            imagesFile: "train-images-idx3-ubyte",
            labelsFile: "train-labels-idx1-ubyte",
            flattening: flattening,
            normalizing: normalizing)
        self.trainImages = trainImages
        self.trainLabels = trainLabels
        self.trainingSize = Int(trainLabels.shape[0])

        let (testImages, testLabels) = readMNIST(
            imagesFile: "t10k-images-idx3-ubyte",
            labelsFile: "t10k-labels-idx1-ubyte",
            flattening: flattening,
            normalizing: normalizing)
        self.testImages = testImages
        self.testLabels = testLabels
        self.testSize = Int(testLabels.shape[0])
    }
}

extension Tensor {
    public func minibatch(at index: Int, batchSize: Int) -> Tensor {
        let start = index * batchSize
        return self[start..<start+batchSize]
    }
}

/// Reads a file into an array of bytes.
func readFile(_ path: String, possibleDirectories: [String]) -> [UInt8] {
    for folder in possibleDirectories {
        let parent = URL(fileURLWithPath: folder)
        let filePath = parent.appendingPathComponent(path)
        guard FileManager.default.fileExists(atPath: filePath.path) else {
            continue
        }
        let data = try! Data(contentsOf: filePath, options: [])
        return [UInt8](data)
    }
    print("File not found: \(path)")
    exit(-1)
}

/// Reads MNIST images and labels from specified file paths.
func readMNIST(imagesFile: String, labelsFile: String, flattening: Bool, normalizing: Bool) -> (
    images: Tensor<Float>,
    labels: Tensor<Int32>
) {
    print("Reading data from files: \(imagesFile), \(labelsFile)")
    let images = readFile(imagesFile, possibleDirectories: [".", "./Datasets/MNIST"]).dropFirst(16)
        .map(Float.init)
    let labels = readFile(labelsFile, possibleDirectories: [".", "./Datasets/MNIST"]).dropFirst(8)
        .map(Int32.init)
    let rowCount = labels.count
    let imageHeight = 28
    let imageWidth = 28

    print("Constructing data tensors.")

    if flattening {
        var flattenedImages = Tensor(shape: [rowCount, imageHeight * imageWidth], scalars: images)
            / 255.0
        if normalizing {
            flattenedImages = flattenedImages * 2.0 - 1.0
        }
        return (images: flattenedImages, labels: Tensor(labels))
    } else {
        return (
            images: Tensor(shape: [rowCount, 1, imageHeight, imageWidth], scalars: images)
                .transposed(withPermutations: [0, 2, 3, 1]) / 255,  // NHWC
            labels: Tensor(labels)
        )
    }
}
