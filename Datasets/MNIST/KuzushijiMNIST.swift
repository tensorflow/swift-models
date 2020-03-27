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
// "KMNIST Dataset" (created by CODH), https://arxiv.org/abs/1812.01718
// adapted from "Kuzushiji Dataset" (created by NIJL and others), doi:10.20676/00000341

import Foundation
import TensorFlow

public struct KuzushijiMNIST: ImageClassificationDataset {
    public let trainingDataset: Dataset<LabeledExample>
    public let testDataset: Dataset<LabeledExample>
    public let trainingExampleCount = 60000
    public let testExampleCount = 10000

    public init() {
        self.init(flattening: false, normalizing: false)
    }

    public init(
        flattening: Bool = false, normalizing: Bool = false,
        localStorageDirectory: URL = DatasetUtilities.defaultDirectory
            .appendingPathComponent("KuzushijiMNIST", isDirectory: true)
    ) {
        self.trainingDataset = Dataset<LabeledExample>(
            elements: fetchMNISTDataset(
                localStorageDirectory: localStorageDirectory,
                remoteBaseDirectory: "http://codh.rois.ac.jp/kmnist/dataset/kmnist/",
                imagesFilename: "train-images-idx3-ubyte",
                labelsFilename: "train-labels-idx1-ubyte",
                flattening: flattening,
                normalizing: normalizing))

        self.testDataset = Dataset<LabeledExample>(
            elements: fetchMNISTDataset(
                localStorageDirectory: localStorageDirectory,
                remoteBaseDirectory: "http://codh.rois.ac.jp/kmnist/dataset/kmnist/",
                imagesFilename: "t10k-images-idx3-ubyte",
                labelsFilename: "t10k-labels-idx1-ubyte",
                flattening: flattening,
                normalizing: normalizing))
    }
}
