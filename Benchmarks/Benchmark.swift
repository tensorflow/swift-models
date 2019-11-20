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
    func run()
}

enum BenchmarkVariety {
    case inferenceThroughput
    case trainingTime
}

struct BenchmarkSettings {
    let batches: Int
    let batchSize: Int
    let iterations: Int
    let epochs: Int
}

struct BenchmarkResults {
    let name: String
    let timings: [Double]
    let settings: BenchmarkSettings
    let variety: BenchmarkVariety
}

extension BenchmarkResults {
    var interpretedTimings: [Double] {
        switch self.variety {
        case .inferenceThroughput:
            let batches = settings.batches
            let batchSize = settings.batchSize
            return timings.map { Double(batches * batchSize) / ($0 / 1000.0) }
        case .trainingTime:
            return timings
        }
    }
}

/// Performs the specified benchmark over a certain number of iterations and provides the result to a callback function.
func benchmark(
    name: String,
    settings: BenchmarkSettings,
    variety: BenchmarkVariety,
    benchmark: Benchmark,
    callback: (BenchmarkResults) -> Void
) {
    var timings: [Double] = []
    for _ in 0..<settings.iterations {
        let timing = time(benchmark.run)
        timings.append(timing)
    }

    let results = BenchmarkResults(
        name: name, timings: timings, settings: settings, variety: variety)
    callback(results)
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

/// Provides the average and standard deviation of an array of values.
func statistics(for values: [Double]) -> (average: Double, standardDeviation: Double) {
    guard values.count > 0 else { return (average: 0.0, standardDeviation: 0.0) }
    guard values.count > 1 else { return (average: values.first!, standardDeviation: 0.0) }

    let average = (values.reduce(0.0) { $0 + $1 }) / Double(values.count)

    let standardDeviation = sqrt(
        values.reduce(0.0) { $0 + ($1 - average) * ($1 - average) }
            / Double(values.count - 1))

    return (average: average, standardDeviation: standardDeviation)
}

// This is a simple callback function example that only logs the result to the console.
func logResults(_ result: BenchmarkResults) {
    let (average, standardDeviation) = statistics(for: result.interpretedTimings)

    switch result.variety {
    case .inferenceThroughput:
        print(
            """
                Benchmark: \(result.name):
                \tAfter \(result.settings.iterations) iterations:
                \tSamples per second: \(String(format: "%.2f", average)), standard deviation: \(String(format: "%.2f", standardDeviation))
            """)
    case .trainingTime:
        print(
            """
                Benchmark: \(result.name):
                \tAfter \(result.settings.iterations) iterations:
                \tAverage: \(String(format: "%.2f", average)) ms, standard deviation: \(String(format: "%.2f", standardDeviation)) ms
            """)
    }
}
