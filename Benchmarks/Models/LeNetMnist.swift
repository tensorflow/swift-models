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

enum LeNetMNIST: BenchmarkModel {
    static var name: String { "LeNetMNIST" }

    static func examplesPerEpoch(for variety: BenchmarkVariety) -> Int {
        switch variety {
        case .inferenceThroughput: return 10000
        case .trainingThroughput: return 60000
        }
    }

    static func defaults(for variety: BenchmarkVariety) -> BenchmarkSettings {
        switch variety {
        case .inferenceThroughput:
            return BenchmarkSettings(
                batches: 1000, batchSize: 128, iterations: 10,
                warmupBatches: 1, synthetic: false, backend: .eager)
        case .trainingThroughput:
            return BenchmarkSettings(
                batches: 110, batchSize: 128, iterations: 1, warmupBatches: 1,
                synthetic: false, backend: .eager)
        }
    }

    static func makeInferenceBenchmark(settings: BenchmarkSettings) -> Benchmark {
        return ImageClassificationInference<LeNet, MNIST>(settings: settings)
    }

    static func makeTrainingBenchmark(settings: BenchmarkSettings) -> Benchmark {
        return ImageClassificationTraining<LeNet, MNIST>(settings: settings)
    }
}

extension LeNet: ImageClassificationModel {
    static var preferredInputDimensions: [Int] { [28, 28, 1] }
    static var outputLabels: Int { 10 }
}
