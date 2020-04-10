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
    var batchSize: Int { get }
    func run() -> [Double]
}

/// Performs the specified benchmark over a certain number of iterations and provides the result to a callback function.
func measure(
    configuration: BenchmarkConfiguration,
    benchmark: Benchmark
) -> BenchmarkResults {
    var timings: [Double] = []
    var warmup: [Double] = []
    let iterations = configuration.settings.iterations
    for _ in 0..<iterations {
        var timedSteps = benchmark.run()
        warmup.append(contentsOf: timedSteps.prefix(configuration.settings.warmupBatches))
        timedSteps.removeFirst(configuration.settings.warmupBatches)
        timings.append(contentsOf: timedSteps)
    }

    return BenchmarkResults(
        configuration: configuration, warmup: warmup, timings: timings,
        batchSize: benchmark.batchSize)
}

// Returns an uptime-based timestamp in milliseconds.
func timestampInMilliseconds() -> Double {
    let divisor: Double = 1_000_000
    return Double(DispatchTime.now().uptimeNanoseconds) / divisor
}

/// Calculates and returns the time in milliseconds since the given timestamp.
func durationInMilliseconds(since timestamp: Double) -> Double {
    let now = timestampInMilliseconds()
    return now - timestamp
}

/// Returns the time elapsed while running `body` in milliseconds.
func time(_ body: () -> Void) -> Double {
    let start = timestampInMilliseconds()
    body()
    return durationInMilliseconds(since: start)
}
