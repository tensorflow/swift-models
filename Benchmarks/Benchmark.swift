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

struct BenchmarkResults: CustomStringConvertible {
    enum BenchmarkResultType {
        case inference(samplesPerSecond: Double, standardDeviation: Double)
        case training(averageTime: Double, standardDeviation: Double)
    }

    let name: String
    let iterations: Int
    let result: BenchmarkResultType

    var description: String {
        switch result {
        case let .inference(samplesPerSecond, standardDeviation):
            return """
                    Benchmark: \(name):
                    \tAfter \(iterations) iterations:
                    \tSamples per second: \(samplesPerSecond), standard deviation: \(standardDeviation)
                """
        case let .training(averageTime, standardDeviation):
            return """
                    Benchmark: \(name):
                    \tAfter \(iterations) iterations:
                    \tAverage: \(averageTime) ms, standard deviation: \(standardDeviation) ms
                """
        }
    }
}

enum BenchmarkVariety {
    case inferenceThroughput(batches: Int, batchSize: Int)
    case trainingTime

    func interpretTiming(_ timing: Double) -> Double {
        switch self {
        case let .inferenceThroughput(batches, batchSize):
            return Double(batches * batchSize) / (timing / 1000.0)
        case .trainingTime:
            return timing
        }
    }
}

/// Performs the specified benchmark over a certain number of iterations and provides the result to a callback function.
func benchmark(
    name: String,
    iterations: Int, variety: BenchmarkVariety, setup: ((BenchmarkVariety) -> Void)? = nil,
    operation: () -> Void,
    callback: (BenchmarkResults) -> Void
) {
    setup?(variety)

    var timings: [Double] = []
    for _ in 0..<iterations {
        let timing = time(operation)
        timings.append(variety.interpretTiming(timing))
    }

    let (samplesPerSecond, standardDeviation) = statistics(for: timings)
    let results = BenchmarkResults(
        name: name, iterations: iterations,
        result: .inference(samplesPerSecond: samplesPerSecond, standardDeviation: standardDeviation)
    )
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
func logResults(results: BenchmarkResults) {
    print("\(results)")
}
