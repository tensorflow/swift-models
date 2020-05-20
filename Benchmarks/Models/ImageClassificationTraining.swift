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

import Batcher
import Datasets
import TensorFlow

// TODO: Ease the tight restriction on Batcher data sources to allow for lazy datasets.
struct ImageClassificationTraining<Model, ClassificationDataset>: Benchmark
where
    Model: ImageClassificationModel, Model.TangentVector.VectorSpaceScalar == Float,
    ClassificationDataset: ImageClassificationData
{
    let duration: BenchmarkDuration
    let batchSize: Int

    init(settings: BenchmarkSettings) {
        self.duration = settings.duration
        self.batchSize = settings.batchSize
    }

    func run(backend: Backend) -> [Double] {
        // Include model and optimizer initialization time in first batch, to be part of warmup.
        // Also include time for following workaround to allocate memory for eager runtime.
        var beforeBatch = timestampInMilliseconds()

        // Note: The this initial eager-mode tensor computation is needed, or all GPU memory
        // will be exhausted on initial allocation of the model.
        // TODO: Remove the following tensor workaround when above is fixed.
        let testTensor = Tensor<Float>([1.0, 2.0, 3.0])
        let testTensor2 = Tensor<Float>([1.0, 2.0, 3.0])
        let _ = testTensor + testTensor2

        let device: Device
        switch backend {
        case .eager: device = Device.defaultTFEager
        case .x10: device = Device.defaultXLA
        }

        var model = Model()
        model.move(to: device)
        // TODO: Split out the optimizer as a separate specification.
        var optimizer = SGD(for: model, learningRate: 0.1)
        optimizer = SGD(copying: optimizer, to: device)
        var batchTimings: [Double] = []
        var currentBatch = 0

        let dataset = ClassificationDataset(batchSize: batchSize, on: device)

        Context.local.learningPhase = .training
        for (epoch, epochBatches) in dataset.training.enumerated() {
            if case let .epochs(epochs) = duration, epoch >= epochs {
                return batchTimings
            }

            for batch in epochBatches {
                let (images, labels) = (batch.data, batch.label)

                let 𝛁model = TensorFlow.gradient(at: model) { model -> Tensor<Float> in
                    let logits = model(images)
                    return softmaxCrossEntropy(logits: logits, labels: labels)
                }
                optimizer.update(&model, along: 𝛁model)
                LazyTensorBarrier()
                batchTimings.append(durationInMilliseconds(since: beforeBatch))
                currentBatch += 1
                if case let .batches(batches) = duration, currentBatch >= batches {
                    return batchTimings
                }
                beforeBatch = timestampInMilliseconds()
            }
        }
        return batchTimings
    }
}
