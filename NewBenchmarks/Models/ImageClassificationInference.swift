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
import ImageClassificationModels
import TensorFlow

protocol ImageClassificationModel: Layer where Input == Tensor<Float>, Output == Tensor<Float> {
  init()
  static var preferredInputDimensions: [Int] { get }
  static var outputLabels: Int { get }
}

// TODO: Ease the tight restriction on Batcher data sources to allow for lazy datasets.
func runImageClassificationInference<Model, ClassificationDataset>(
  model modelType: Model.Type,
  dataset datasetType: ClassificationDataset.Type,
  state: inout BenchmarkState
) throws
where
  Model: ImageClassificationModel,
  ClassificationDataset: ImageClassificationData
{
  let settings = state.settings
  let device = settings.device
  let batchSize = settings.batchSize!
  let dataset = ClassificationDataset(batchSize: batchSize, on: device)
  var model = Model()
  model.move(to: device)

  for epochBatches in dataset.training {
    for batch in epochBatches {
      let images = batch.data

      do {
        try state.measure {
          let _ = model(images)
          LazyTensorBarrier()
        }
      } catch {
        if settings.backend == .x10 {
          // A synchronous barrier is needed for X10 to ensure all execution completes
          // before tearing down the model.
          LazyTensorBarrier(wait: true)
        }
        throw error
      }
    }
  }
}
