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
// "The MNIST database of handwritten digits"
// Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
// http://yann.lecun.com/exdb/mnist/

import Foundation
import TensorFlow

public struct MNIST: ImageClassificationDataset {
    public let trainingDataset: Dataset<LabeledExample>
    public let testDataset: Dataset<LabeledExample>
    public let trainingExampleCount = 60000

    public init() {
        self.init(flattening: false, normalizing: false)
    }

    public init(
        flattening: Bool = false, normalizing: Bool = false,
        localStorageDirectory: URL = DatasetUtilities.currentWorkingDirectoryURL
    ) {
        self.trainingDataset = Dataset<LabeledExample>(
            elements: fetchDataset(
                localStorageDirectory: localStorageDirectory,
                imagesFilename: "train-images-idx3-ubyte",
                labelsFilename: "train-labels-idx1-ubyte",
                flattening: flattening,
                normalizing: normalizing))

        self.testDataset = Dataset<LabeledExample>(
            elements: fetchDataset(
                localStorageDirectory: localStorageDirectory,
                imagesFilename: "t10k-images-idx3-ubyte",
                labelsFilename: "t10k-labels-idx1-ubyte",
                flattening: flattening,
                normalizing: normalizing))
    }
}

fileprivate func fetchDataset(
    localStorageDirectory: URL,
    imagesFilename: String,
    labelsFilename: String,
    flattening: Bool,
    normalizing: Bool
) -> LabeledExample {
    guard let remoteRoot:URL = URL(string: "http://yann.lecun.com/exdb/mnist") else {
        fatalError("Failed to create MNST root url: http://yann.lecun.com/exdb/mnist")
    }

    let imagesData = DatasetUtilities.fetchResource(
        filename: imagesFilename,
        remoteRoot: remoteRoot,
        localStorageDirectory: localStorageDirectory)
    let labelsData = DatasetUtilities.fetchResource(
        filename: labelsFilename,
        remoteRoot: remoteRoot,
        localStorageDirectory: localStorageDirectory)

    let images = [UInt8](imagesData).dropFirst(16).map(Float.init)
    let labels = [UInt8](labelsData).dropFirst(8).map(Int32.init)

    let rowCount = labels.count
    let (imageWidth, imageHeight) = (28, 28)

    if flattening {
        var flattenedImages = Tensor(shape: [rowCount, imageHeight * imageWidth], scalars: images)
            / 255.0
        if normalizing {
            flattenedImages = flattenedImages * 2.0 - 1.0
        }
        return LabeledExample(label: Tensor(labels), data: flattenedImages)
    } else {
        return LabeledExample(
            label: Tensor(labels),
            data:
                Tensor(shape: [rowCount, 1, imageHeight, imageWidth], scalars: images)
                .transposed(permutation: [0, 2, 3, 1]) / 255  // NHWC
        )
    }
}
