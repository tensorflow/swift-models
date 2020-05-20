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

enum PrintingStyle {
    case plainText
    case json
}

extension BenchmarkResults {
    func print(using style: PrintingStyle) {
        switch style {
        case .plainText:
            printAsPlainText()
        case .json:
            printAsJSON()
        }
    }

    private func printAsPlainText() {
        let configuration = self.configuration
        let settings = configuration.settings
        
        Swift.print("Benchmark: \(configuration.name)")
        Swift.print("\tVariety: \(configuration.variety.rawValue)")
        Swift.print("\tAfter \(settings.iterations) iteration(s) of \(self.timings.count) batches:")
        let timeAfterWarmup = self.totalTime - self.warmupTime
        let totalExamples = self.timings.count * settings.batchSize
        let averageExamplesPerSecond = Double(totalExamples) / (Double(timeAfterWarmup) / 1000.0)
        Swift.print(
            "\tAverage examples per second: \(String(format: "%.2f", averageExamplesPerSecond))")
        Swift.print("\tInstantaneous examples per second:")
        let examplesPerSecond = self.timings.map {
            Double(settings.batchSize) / ($0 / 1000.0)
        }
        examplesPerSecond.printStatistics(indentation: 2)
        Swift.print("\tStep time (ms):")
        self.timings.printStatistics(indentation: 2)
        Swift.print("\tTotal time: \(String(format: "%.2f", self.totalTime)) ms")
        Swift.print("\tWarmup time: \(String(format: "%.2f", self.warmupTime)) ms")
        Swift.print("\tTime after warmup: \(String(format: "%.2f", timeAfterWarmup)) ms")
    }

    private func printAsJSON() {
        printJSON(self)
    }
}

extension BenchmarkConfiguration {
    func print(using style: PrintingStyle) {
        switch style {
        case .plainText:
            printAsPlainText()
        case .json:
            printAsJSON()
        }
    }

    private func printAsPlainText() {
        var result = ""
        result += "--benchmark "
        result += self.name
        result += " "
        switch self.variety {
        case .trainingThroughput:
            result += "--training "
        case .inferenceThroughput:
            result += "--inference "
        }
        result += "--\(settings.backend) "
        if self.settings.synthetic {
            result += "--synthetic "
        } else {
            result += "--real "
        }
        switch settings.duration {
        case let .batches(value):
            result += "--batches \(value) "
        case let .epochs(value):
            result += "--epochs \(value) "
        }
        result += "--batchSize \(settings.batchSize) "
        result += "--iterations \(settings.iterations) "
        result += "--warmupBatches \(settings.warmupBatches) "
        Swift.print(result)
    }

    private func printAsJSON() {
        printJSON(self)
    }
}

/// Print given encodable value to stdout.
func printJSON<T: Encodable>(_ value: T) {
    let data = try! JSONEncoder().encode(value)
    let json = String(data: data, encoding: .utf8)!
    print(json)
}

/// Statistics-related helpers.
extension Array where Element == Double {
    /// The average value of elements.
    var average: Element {
        guard count > 0 else { return 0 }
        guard count > 1 else { return first! }
        return (reduce(into: 0) { $0 += $1 }) / Double(count)
    }

    /// The variance of elements.
    var variance: Element {
        guard count > 0 else { return 0 }
        guard count > 1 else { return 0 }
        let deltaDegreesOfFreedom = 1
        let squaredDeviationsSum = reduce(into: 0) { $0 += ($1 - average) * ($1 - average) }
        return squaredDeviationsSum / Double(count - deltaDegreesOfFreedom)
    }

    /// The standard deviation of elements.
    var standardDeviation: Element {
        return Double.sqrt(variance)
    }
    
    func printStatistics(indentation: Int) {
        let tabs = String(repeating: "\t", count: indentation)
        Swift.print("\(tabs)Average: \(String(format: "%.2f", self.average))")
        Swift.print("\(tabs)Min: \(String(format: "%.2f", self.min()!))")
        Swift.print("\(tabs)Max: \(String(format: "%.2f", self.max()!))")
        Swift.print("\(tabs)Standard deviation: \(String(format: "%.2f", self.standardDeviation))")
    }
}
