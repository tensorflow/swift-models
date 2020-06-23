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

import Datasets
import ImageClassificationModels
import TensorFlow

protocol ImageClassificationModel: Layer where Input == Tensor<Float>, Output == Tensor<Float> {
    init()
    static var preferredInputDimensions: [Int] { get }
    static var outputLabels: Int { get }
}

class ImageClassificationInference<Model, ClassificationDataset>: Benchmark
where
    Model: ImageClassificationModel,
    ClassificationDataset: ImageClassificationData
{
    var model: Model
    let duration: BenchmarkDuration
    let batchSize: Int

    init(settings: BenchmarkSettings) {
        self.duration = settings.duration
        self.batchSize = settings.batchSize
        self.model = Model()
    }

    func run(backend: Backend) -> [Double] {
        let device: Device
        switch backend {
        case .eager: device = Device.defaultTFEager
        case .x10: device = Device.defaultXLA
        }
        let dataset = ClassificationDataset(batchSize: batchSize, on: device)

        model.move(to: device)

        var batchTimings: [Double] = []
        var currentBatch = 0

        for (epoch, epochBatches) in dataset.training.enumerated() {
            if case let .epochs(epochs) = duration, epoch >= epochs {
                if backend == .x10 {
                    // A synchronous barrier is needed for X10 to ensure all execution completes
                    // before tearing down the model.
                    LazyTensorBarrier(wait: true)
                }
                return batchTimings
            }

            for batch in epochBatches {
                let images = batch.data

                let batchTime = time {
                    let _ = model(images)
                    LazyTensorBarrier()
                }
                batchTimings.append(batchTime)
                currentBatch += 1
                if case let .batches(batches) = duration, currentBatch >= batches {
                    if backend == .x10 {
                        // A synchronous barrier is needed for X10 to ensure all execution completes
                        // before tearing down the model.
                        LazyTensorBarrier(wait: true)
                    }
                    return batchTimings
                }
            }
        }
        if backend == .x10 {
            // A synchronous barrier is needed for X10 to ensure all execution completes
            // before tearing down the model.
            LazyTensorBarrier(wait: true)
        }
        return batchTimings
    }
}
