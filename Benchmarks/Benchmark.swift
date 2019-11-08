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

protocol BenchmarkResults: CustomStringConvertible {
    var description: String { get }
}

struct TrainingBenchmarkResults: BenchmarkResults {
    let iterations: Int
    let averageTime: Double
    let standardDeviation: Double

    var description: String {
        get {
            return """
                  \tAfter \(iterations) iterations:
                  \tAverage: \(averageTime) ms, standard deviation: \(standardDeviation) ms
                """
        }
    }
}

struct InferenceBenchmarkResults: BenchmarkResults {
    let iterations: Int
    let samplesPerSecond: Double
    let standardDeviation: Double

    var description: String {
        get {
            return """
                  \tAfter \(iterations) iterations:
                  \tSamples per second: \(samplesPerSecond), standard deviation: \(standardDeviation)
                """
        }
    }
}

func timeExecution(_ operation: () -> Void) -> Double {
    var startTime = timeval()
    gettimeofday(&startTime, nil)

    operation()

    var endTime = timeval()
    gettimeofday(&endTime, nil)
    let secondsPortion = (endTime.tv_sec - startTime.tv_sec) * 1000
    let microsecondsPortion = Int((endTime.tv_usec - startTime.tv_usec) / 1000)
    return Double(secondsPortion + microsecondsPortion)
}

func statistics(for values: [Double]) -> (average: Double, standardDeviation: Double) {
    guard values.count > 0 else { return (average: 0.0, standardDeviation: 0.0) }
    guard values.count > 1 else { return (average: values.first!, standardDeviation: 0.0) }

    let average = (values.reduce(0.0) { $0 + $1 }) / Double(values.count)

    let standardDeviation = sqrt(
        values.reduce(0.0) { $0 + ($1 - average) * ($1 - average) }
            / Double(values.count - 1))

    return (average: average, standardDeviation: standardDeviation)
}

func benchmarkTraining(iterations: Int, operation: () -> Void, callback: (BenchmarkResults) -> Void)
{
    var timings: [Double] = []
    for _ in 0..<iterations {
        timings.append(timeExecution(operation))
    }

    let (averageTime, standardDeviation) = statistics(for: timings)

    let results = TrainingBenchmarkResults(
        iterations: iterations, averageTime: averageTime, standardDeviation: standardDeviation)
    callback(results)
}

func benchmarkInference(
    iterations: Int, batches: Int, batchSize: Int, setup: (Int, Int) -> Void, operation: () -> Void,
    callback: (BenchmarkResults) -> Void
) {
    setup(batches, batchSize)

    var timings: [Double] = []
    for _ in 0..<iterations {
        timings.append(Double(batches * batchSize) / (timeExecution(operation) / 1000.0))
    }

    let (samplesPerSecond, standardDeviation) = statistics(for: timings)
    let results = InferenceBenchmarkResults(
        iterations: iterations, samplesPerSecond: samplesPerSecond,
        standardDeviation: standardDeviation)
    callback(results)
}

func logResults(name: String) -> (BenchmarkResults) -> Void {
    return { results in
        print("Benchmark: \(name):")
        print("\(results)")
    }
}
