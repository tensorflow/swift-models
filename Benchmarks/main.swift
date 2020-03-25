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

import ArgumentParser
import ModelSupport

/// Collect benchmark results and print them to stdout using a given printing style
func runBenchmark(
    named name: String,
    settings: BenchmarkSettings,
    variety: BenchmarkVariety,
    style: PrintingStyle
) {
    if let benchmarkModel = benchmarkModels[name] {
        var bench: Benchmark
        var benchSettings: BenchmarkSettings
        switch variety {
        case .inferenceThroughput:
            benchSettings =
                settings.withDefaults(benchmarkModel.defaultInferenceSettings)
            bench = benchmarkModel.makeInferenceBenchmark(settings: benchSettings)
        case .trainingTime:
            benchSettings =
                settings.withDefaults(benchmarkModel.defaultTrainingSettings)
            bench = benchmarkModel.makeTrainingBenchmark(settings: benchSettings)
        }
        let configuration = BenchmarkConfiguration(
            name: name,
            variety: variety,
            settings: benchSettings)
        let results = measure(
            configuration: configuration,
            benchmark: bench)
        results.print(using: style)
    } else {
        printError("No registered inference benchmark with a name: \(name)")
        printError("Consider running `list` command to see all available benchmarks.")
    }
}

struct BenchmarkCommand: ParsableCommand {
    static var configuration = CommandConfiguration(
        commandName: "Benchmarks",
        abstract: """
            Runs series of benchmarks against a variety of models
            in the swift-models repository.
            """,
        subcommands: [
            ListDefaultsSubcommand.self,
            MeasureSubcommand.self,
            MeasureAllSubCommand.self
        ])
}

extension BenchmarkCommand {
    struct ListDefaultsSubcommand: ParsableCommand {
        static var configuration = CommandConfiguration(
            commandName: "list-defaults",
            abstract: "List all available benchmarks and their default settings.")

        @Flag(name: .customLong("json"), help: "Output json instead of plain text.")
        var useJSON: Bool

        func run() throws {
            let style: PrintingStyle = useJSON ? .json : .plainText
            for (name, benchModel) in benchmarkModels {
                let trainingConfiguration =
                    BenchmarkConfiguration(
                        name: name,
                        variety: .trainingTime,
                        settings: benchModel.defaultTrainingSettings)
                trainingConfiguration.print(using: style)
                let inferenceConfiguration =
                    BenchmarkConfiguration(
                        name: name,
                        variety: .inferenceThroughput,
                        settings: benchModel.defaultInferenceSettings)
                inferenceConfiguration.print(using: style)
            }
        }
    }
}

extension BenchmarkCommand {
    struct MeasureSubcommand: ParsableCommand {
        static var configuration = CommandConfiguration(
            commandName: "measure",
            abstract: "Run a single benchmark with provided settings."
        )

        @Flag(help: "Run training benchmark.")
        var training: Bool

        @Flag(help: "Run inference benchmark.")
        var inference: Bool

        @Option(default: "", help: "Name of the benchmark to run.")
        var benchmark: String

        @Option(default: -1, help: "Number of batches.")
        var batches: Int

        @Option(name: .customLong("batchSize"), default: -1, help: "Size of a single batch.")
        var batchSize: Int

        @Option(default: -1, help: "Number of benchmark iterations.")
        var iterations: Int

        @Option(default: -1, help: "Number of training epochs.")
        var epochs: Int

        @Flag(name: .customLong("json"), help: "Output json instead of plain text.")
        var useJSON: Bool

        func validate() throws {
            guard training || inference else {
                throw ValidationError(
                    "Must specify either --training or --inference benchmark variety.")
            }

            guard !(training && inference) else {
                throw ValidationError(
                    "Can't specify both --training and --inference benchmark variety.")
            }

            guard !benchmark.isEmpty else {
                throw ValidationError("Must provide a --benchmark to run.")
            }
        }

        func run() throws {
            let style: PrintingStyle = useJSON ? .json : .plainText
            let settings = BenchmarkSettings(
                batches: batches,
                batchSize: batchSize,
                iterations: iterations,
                epochs: epochs)

            let variety: BenchmarkVariety = training ? .trainingTime : .inferenceThroughput
            runBenchmark(
                named: benchmark,
                settings: settings,
                variety: variety,
                style: style
            )
        }
    }
}

extension BenchmarkCommand {
    struct MeasureAllSubCommand: ParsableCommand {
        static var configuration = CommandConfiguration(
            commandName: "measure-all",
            abstract: "Run all benchmarks with default settings."
        )

        @Flag(name: .customLong("json"),
        help: "Output json instead of plain text.")
        var useJSON: Bool

        func run() throws {
            let style: PrintingStyle = useJSON ? .json : .plainText
            for (name, benchmarkModel) in benchmarkModels {
                runBenchmark(
                    named: name,
                    settings: benchmarkModel.defaultTrainingSettings,
                    variety: .trainingTime,
                    style: style)
                runBenchmark(
                    named: name,
                    settings: benchmarkModel.defaultInferenceSettings,
                    variety: .inferenceThroughput,
                    style: style)
            }
        }
    }
}

BenchmarkCommand.main()
