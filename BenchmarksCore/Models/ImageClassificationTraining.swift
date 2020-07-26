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

import Benchmark
import Datasets
import TensorFlow

func runImageClassificationTraining<Model, ClassificationDataset>(
  model modelType: Model.Type,
  dataset datasetType: ClassificationDataset.Type,
  state: inout BenchmarkState
) throws
where
  Model: ImageClassificationModel, Model.TangentVector.VectorSpaceScalar == Float,
  ClassificationDataset: ImageClassificationData
{
  // Include model and optimizer initialization time in first batch, to be part of warmup.
  // Also include time for following workaround to allocate memory for eager runtime.
  state.start()

  let settings = state.settings
  let device = settings.device
  let batchSize = settings.batchSize!
  var model = Model()
  model.move(to: device)
  // TODO: Split out the optimizer as a separate specification.
  var optimizer = SGD(for: model, learningRate: 0.1)
  optimizer = SGD(copying: optimizer, to: device)

  let dataset = ClassificationDataset(batchSize: batchSize, on: device)

  Context.local.learningPhase = .training
  for epochBatches in dataset.training {
    for batch in epochBatches {
      let (images, labels) = (batch.data, batch.label)

      let 𝛁model = TensorFlow.gradient(at: model) { model -> Tensor<Float> in
        let logits = model(images)
        return softmaxCrossEntropy(logits: logits, labels: labels)
      }
      optimizer.update(&model, along: 𝛁model)
      LazyTensorBarrier()
      do {
        try state.end()
      } catch {
        if settings.backend == .x10 {
          // A synchronous barrier is needed for X10 to ensure all execution completes
          // before tearing down the model.
          LazyTensorBarrier(wait: true)
        }
        throw error
      }
      state.start()
    }
  }
}
