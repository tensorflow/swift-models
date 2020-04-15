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

import Datasets
import ImageClassificationModels
import TensorFlow

struct ResNetCIFAR10: BenchmarkModel {

    var defaultInferenceSettings: BenchmarkSettings {
        return BenchmarkSettings(batches: 1000, batchSize: 1, iterations: 10, epochs: -1)
    }

    func makeInferenceBenchmark(settings: BenchmarkSettings) -> Benchmark {
        return ImageClassificationInference<ResNet56, CIFAR10>(settings: settings)
    }

    var defaultTrainingSettings: BenchmarkSettings {
        return BenchmarkSettings(batches: -1, batchSize: 128, iterations: 10, epochs: 1)
    }

    func makeTrainingBenchmark(settings: BenchmarkSettings) -> Benchmark {
        return ImageClassificationTraining<ResNet56, CIFAR10>(settings: settings)
    }
}

struct ResNet56: Layer {
    var model: ResNet
    
    init() {
        model = ResNet(classCount: 10, depth: .resNet56, downsamplingInFirstStage: false)
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return model(input)
    }
}
