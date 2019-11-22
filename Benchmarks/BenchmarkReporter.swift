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

/// Protocol that defines how are the benchmark results printed to the stdout. 
protocol BenchmarkReporter {

    /// Print benchmark results to stdout.
    func printResults(_ result: BenchmarkResults)

    /// Print benchmark default settings to stdout.
    func printDefaults()
}

/// Reporter that prints results as human-readable plain text.
struct PlainTextReporter: BenchmarkReporter {

    func printResults(_ result: BenchmarkResults) {
        let (average, standardDeviation) = statistics(for: result.interpretedTimings)

        switch result.variety {
        case .inferenceThroughput:
            print("Benchmark: \(result.name):")
            print("\tAfter \(result.settings.iterations) iterations:")
            print(
                "\tSamples per second: \(String(format: "%.2f", average)), standard deviation: \(String(format: "%.2f", standardDeviation))"
            )
        case .trainingTime:
            print("Benchmark: \(result.name):")
            print("\tAfter \(result.settings.iterations) iterations:")
            print(
                "\tAverage: \(String(format: "%.2f", average)) ms, standard deviation: \(String(format: "%.2f", standardDeviation)) ms"
            )
        }
    }

    func printDefaults() {
        for (name, benchmarkModel) in benchmarkModels {
            let trainingFlags = flags(from: benchmarkModel.defaultTrainingSettings)
            print("--benchmark \(name) --training \(trainingFlags)")
            let inferenceFlags = flags(from: benchmarkModel.defaultInferenceSettings)
            print("--benchmark \(name) --inference \(inferenceFlags)")
        }
    }
}

/// Auxiliary struct used to show benchmark default settings coupled with benchmark name.
struct JsonDefaults: Codable {
    let name: String
    let variety: BenchmarkVariety
    let settings: BenchmarkSettings
}

/// Reporter that prints results as machine-readable json.
struct JsonReporter: BenchmarkReporter {

    func printJson<T: Encodable>(_ value: T) {
        let data = try! JSONEncoder().encode(value)
        let json = String(data: data, encoding: .utf8)!
        print(json)
    }

    func printResults(_ result: BenchmarkResults) {
        printJson(result)
    }

    func printDefaults(name: String, variety: BenchmarkVariety, settings: BenchmarkSettings) {
        printJson(JsonDefaults(name: name, variety: variety, settings: settings))
    }

    func printDefaults() {
        for (name, benchmarkModel) in benchmarkModels {
            printDefaults(
                name: name,
                variety: .trainingTime,
                settings: benchmarkModel.defaultTrainingSettings)
            printDefaults(
                name: name,
                variety: .inferenceThroughput,
                settings: benchmarkModel.defaultInferenceSettings)
        }
    }
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

/// Represent default settings as a string that contains equivalent benchmark flags.
func flags(from settings: BenchmarkSettings) -> String {
    var result = ""
    if settings.batches != -1 {
        result += "--batches \(settings.batches) "
    }
    if settings.batchSize != -1 {
        result += "--batchSize \(settings.batchSize) "
    }
    if settings.iterations != -1 {
        result += "--iterations \(settings.iterations) "
    }
    if settings.epochs != -1 {
        result += "--epochs \(settings.epochs) "
    }
    return result
}
