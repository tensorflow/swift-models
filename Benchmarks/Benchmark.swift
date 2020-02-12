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

import Foundation

protocol Benchmark {
    var exampleCount: Int { get }
    func run()
}

/// Performs the specified benchmark over a certain number of iterations and provides the result to a callback function.
func measure(
    configuration: BenchmarkConfiguration,
    benchmark: Benchmark
) -> BenchmarkResults {
    var timings: [Double] = []
    for _ in 0..<configuration.settings.iterations {
        let timing = time(benchmark.run)
        timings.append(timing)
    }

    return BenchmarkResults(
        configuration: configuration, timings: timings, exampleCount: benchmark.exampleCount)
}

/// Returns the time elapsed while running `body` in milliseconds.
func time(_ body: () -> Void) -> Double {
    let divisor: Double = 1_000_000
    let start = Double(DispatchTime.now().uptimeNanoseconds) / divisor
    body()
    let end = Double(DispatchTime.now().uptimeNanoseconds) / divisor
    let elapsed = end - start
    return elapsed
}
