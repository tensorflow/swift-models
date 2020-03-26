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
// "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms"
// Han Xiao and Kashif Rasul and Roland Vollgraf
// https://arxiv.org/abs/1708.07747

import Foundation
import TensorFlow
import Batcher

public struct FashionMNIST: ImageClassificationDataset {
    public typealias SourceDataSet = [TensorPair<Float, Int32>]
    public let trainingBatcher: Batcher<SourceDataSet>
    public let testBatcher: Batcher<SourceDataSet>

    public init(batchSize: Int) {
        self.init(batchSize: batchSize, flattening: false, normalizing: false)
    }

    public init(
        batchSize: Int, flattening: Bool = false, normalizing: Bool = false,
        localStorageDirectory: URL = FileManager.default.temporaryDirectory.appendingPathComponent(
            "FashionMNIST", isDirectory: true)
    ) {
        trainingBatcher = Batcher<SourceDataSet>(
            on: fetchMNISTDataset(
                localStorageDirectory: localStorageDirectory,
                remoteBaseDirectory: "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
                imagesFilename: "train-images-idx3-ubyte",
                labelsFilename: "train-labels-idx1-ubyte",
                flattening: flattening,
                normalizing: normalizing),
            batchSize: batchSize,
            numWorkers: 1, //No need to use parallelism since everything is loaded in memory
            shuffle: true)

        testBatcher = Batcher<SourceDataSet>(
            on: fetchMNISTDataset(
                localStorageDirectory: localStorageDirectory,
                remoteBaseDirectory: "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
                imagesFilename: "t10k-images-idx3-ubyte",
                labelsFilename: "t10k-labels-idx1-ubyte",
                flattening: flattening,
                normalizing: normalizing),
            batchSize: batchSize,
            numWorkers: 1) //No need to use parallelism since everything is loaded in memory
    }
}
