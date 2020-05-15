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

// This benchmark currently is purely synthetic, as we don't yet have ImageNet as a real dataset.
enum ResNetImageNet: BenchmarkModel {
    static var name: String { "ResNetImageNet" }

    static func defaults(for variety: BenchmarkVariety) -> BenchmarkSettings {
        switch(variety) {
        case .inferenceThroughput:
            return BenchmarkSettings(duration: .batches(1000), batchSize: 128, iterations: 10,
                warmupBatches: 1, synthetic: true, backend: .eager)
        case .trainingThroughput:
            return BenchmarkSettings(duration: .batches(110), batchSize: 128, iterations: 1,
                warmupBatches: 1, synthetic: true, backend: .eager)
        }
    }

    // Note: using a purely synthetic dataset in all cases until ImageNet is formally implemented.
    static func makeInferenceBenchmark(settings: BenchmarkSettings) -> Benchmark {
        return ImageClassificationInference<ResNet50, SyntheticImageNet>(settings: settings)
    }

    static func makeTrainingBenchmark(settings: BenchmarkSettings) -> Benchmark {
        return ImageClassificationTraining<ResNet50, SyntheticImageNet>(settings: settings)
    }
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
