// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

import Batcher
import Datasets
import TensorFlow

public struct SyntheticImageDataset: ImageClassificationDataset {
    public typealias SourceDataSet = [TensorPair<Float, Int32>]
    public let training: Batcher<SourceDataSet>
    public let test: Batcher<SourceDataSet>

    public init(batchSize: Int) {
        self.init(batchSize: batchSize, batches: 110, labels: 10, dimensions: [224, 224, 3])
    }

    // TODO: Use a random seed to create deterministic examples here.
    public init(batchSize: Int, batches: Int, labels: Int, dimensions: [Int]) {
        precondition(labels > 0)
        precondition(dimensions.count == 3)
        let totalExamples = batchSize * batches

        let syntheticDataset = (0..<totalExamples).map { _ -> TensorPair<Float, Int32> in
            let syntheticImage = Tensor<Float>(
                randomNormal: TensorShape(dimensions), mean: Tensor<Float>(0.5),
                standardDeviation: Tensor<Float>(0.1))
            let syntheticLabel = Tensor<Int32>(Int32.random(in: 0..<Int32(labels)))
            return TensorPair(first: syntheticImage, second: syntheticLabel)
        }

        training = Batcher<SourceDataSet>(
            on: syntheticDataset, batchSize: batchSize, numWorkers: 1, shuffle: true)

        test = Batcher<SourceDataSet>(on: syntheticDataset, batchSize: batchSize, numWorkers: 1)
    }
}
