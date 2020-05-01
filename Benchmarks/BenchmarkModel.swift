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

/// Protocol that contains benchmark factory methods for a given model. 
protocol BenchmarkModel {
    /// A string identifier for this model, to be used in logging and at the command line.
    static var name: String { get }

    /// The number of examples per epoch in the training or test dataset for this benchmark.
    static func examplesPerEpoch(for variety: BenchmarkVariety) -> Int

    /// Settings to use in a specific benchmark if no custom flags are given.
    static func defaults(for variety: BenchmarkVariety) -> BenchmarkSettings

    /// Create an instance of inference  benchmark with given settings.
    static func makeInferenceBenchmark(settings: BenchmarkSettings) -> Benchmark

    /// Create an instance of training benchmark with given settings.
    static func makeTrainingBenchmark(settings: BenchmarkSettings) -> Benchmark
}
