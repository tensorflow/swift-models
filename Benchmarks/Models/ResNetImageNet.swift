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

import Benchmark
import Datasets
import ImageClassificationModels
import TensorFlow

let ResNetImageNet = BenchmarkSuite(
  name: "ResNetImageNet",
  settings: BatchSize(128), WarmupIterations(1), Synthetic(true)
) { suite in

  func inference(state: inout BenchmarkState) throws {
    if state.settings.synthetic! {
      try runImageClassificationInference(
        model: ResNet50.self, dataset: SyntheticImageNet.self, state: &state)
    } else {
      fatalError("Only synthetic ImageNet benchmarks are supported at the moment.")
    }
  }

  func training(state: inout BenchmarkState) throws {
    if state.settings.synthetic! {
      try runImageClassificationTraining(
        model: ResNet50.self, dataset: SyntheticImageNet.self, state: &state)
    } else {
      fatalError("Only synthetic ImageNet benchmarks are supported at the moment.")
    }
  }

  suite.benchmark("inference", settings: Backend(.eager), function: inference)
  suite.benchmark("inference_x10", settings: Backend(.x10), function: inference)
  suite.benchmark("training", settings: Backend(.eager), function: training)
  suite.benchmark("training_x10", settings: Backend(.eager), function: training)
}

struct ResNet50: Layer {
  var model: ResNet

  init() {
    model = ResNet(classCount: 1000, depth: .resNet50, downsamplingInFirstStage: true)
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    return model(input)
  }
}

extension ResNet50: ImageClassificationModel {
  static var preferredInputDimensions: [Int] { [224, 224, 3] }
  static var outputLabels: Int { 1000 }
}

final class SyntheticImageNet: SyntheticImageDataset<SystemRandomNumberGenerator>,
  ImageClassificationData
{
  public init(batchSize: Int, on device: Device = Device.default) {
    super.init(
      batchSize: batchSize, labels: ResNet50.outputLabels,
      dimensions: ResNet50.preferredInputDimensions, entropy: SystemRandomNumberGenerator(),
      device: device)
  }
}
