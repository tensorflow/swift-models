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

    /// Settings to use in inference benchmark if no custom flags are given.
    var defaultInferenceSettings: BenchmarkSettings { get }

    /// Create an instance of inference  benchmark with given settings.
    func makeInferenceBenchmark(_ settings: BenchmarkSettings) -> Benchmark

    /// Settings to use in training benchmark if no custom flags are given.
    var defaultTrainingSettings: BenchmarkSettings { get }

    /// Create an instance of training benchmark with given settings.
    func makeTrainingBenchmark(_ settings: BenchmarkSettings) -> Benchmark
}
