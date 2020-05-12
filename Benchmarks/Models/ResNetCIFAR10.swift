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

enum ResNetCIFAR10: BenchmarkModel {
    static var name: String { "ResNetCIFAR10" }

    static func examplesPerEpoch(for variety: BenchmarkVariety) -> Int {
        switch(variety) {
        case .inferenceThroughput: return 10000
        case .trainingThroughput: return 50000
        }
    }

    static func defaults(for variety: BenchmarkVariety) -> BenchmarkSettings {
        switch(variety) {
        case .inferenceThroughput:
            return BenchmarkSettings(batches: 1000, batchSize: 128, iterations: 10,
                                     warmupBatches: 1, synthetic: false, backend: .eager)
        case .trainingThroughput:
            return BenchmarkSettings(batches: 110, batchSize: 128, iterations: 1, warmupBatches: 1,
                                     synthetic: false, backend: .eager)
        }
    }

    static func makeInferenceBenchmark(settings: BenchmarkSettings) -> Benchmark {
        if settings.synthetic {
            return ImageClassificationInference<ResNet56, SyntheticCIFAR10>(settings: settings)
        } else {
            return ImageClassificationInference<ResNet56, CIFAR10<SystemRandomNumberGenerator>>(settings: settings)
        }
    }

    static func makeTrainingBenchmark(settings: BenchmarkSettings) -> Benchmark {
        if settings.synthetic {
            return ImageClassificationTraining<ResNet56, SyntheticCIFAR10>(settings: settings)
        } else {
            return ImageClassificationTraining<ResNet56, CIFAR10<SystemRandomNumberGenerator>>(settings: settings)
        }
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

extension ResNet56: ImageClassificationModel {
    static var preferredInputDimensions: [Int] { [32, 32, 3] }
    static var outputLabels: Int { 10 }
}

final class SyntheticCIFAR10: SyntheticImageDataset<SystemRandomNumberGenerator>, ImageClassificationData {
  public init(batchSize: Int, on device: Device = Device.default) {
    super.init(batchSize: batchSize, labels: ResNet56.outputLabels,
      dimensions: ResNet56.preferredInputDimensions, entropy: SystemRandomNumberGenerator(),
      device: device)
  }
}
