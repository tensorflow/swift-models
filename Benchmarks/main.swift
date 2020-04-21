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
    _ benchmarkModel: BenchmarkModel.Type,
    settings: BenchmarkSettings,
    variety: BenchmarkVariety,
    style: PrintingStyle
) {
    var benchmark: Benchmark
    switch variety {
    case .inferenceThroughput:
        benchmark = benchmarkModel.makeInferenceBenchmark(settings: settings)
    case .trainingThroughput:
        benchmark = benchmarkModel.makeTrainingBenchmark(settings: settings)
    }
    let configuration = BenchmarkConfiguration(
        name: benchmarkModel.name,
        variety: variety,
        settings: settings)
    let results = measure(
        configuration: configuration,
        benchmark: benchmark)
    results.print(using: style)
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
                        variety: .trainingThroughput,
                        settings: benchModel.defaults(for: .trainingThroughput))
                trainingConfiguration.print(using: style)
                let inferenceConfiguration =
                    BenchmarkConfiguration(
                        name: name,
                        variety: .inferenceThroughput,
                        settings: benchModel.defaults(for: .inferenceThroughput))
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

        @Flag(help: "Use synthetic data.")
        var synthetic: Bool

        @Flag(help: "Use real data.")
        var real: Bool

        @Option(help: "Name of the benchmark to run.")
        var benchmark: String?

        @Option(help: "Number of batches.")
        var batches: Int?

        @Option(name: .customLong("batchSize"), help: "Size of a single batch.")
        var batchSize: Int?

        @Option(help: "Number of benchmark iterations.")
        var iterations: Int?

        @Option(help: "Number of training epochs.")
        var epochs: Int?

        @Option(name: .customLong("warmupBatches"),
                help: "Number of batches to use as a warmup period.")
        var warmupBatches: Int?

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
            
            guard !(real && synthetic) else {
                throw ValidationError(
                    "Can't specify both --real and --synthetic data sources.")
            }

            guard !((batches != nil) && (epochs != nil)) else {
                throw ValidationError(
                    "Can't specify both the number of batches and epochs.")
            }

            guard benchmark != nil else {
                throw ValidationError("Must provide a --benchmark to run.")
            }
            
            guard benchmarkModels[benchmark!] != nil else {
                throw ValidationError(
                    "No registered inference benchmark with a name: \(benchmark!). Consider running the `list` command to see all available benchmarks.")
            }
        }

        func run() throws {
            let style: PrintingStyle = useJSON ? .json : .plainText
            let variety: BenchmarkVariety = training ? .trainingThroughput : .inferenceThroughput
            let benchmarkModel = benchmarkModels[benchmark!]!
            let defaults = benchmarkModel.defaults(for: variety)
            let batchSizeToUse = batchSize ?? defaults.batchSize
            let specifiedBatches: Int?
            if let epochs = epochs {
                specifiedBatches = epochs * (benchmarkModel.examplesPerEpoch(for: variety)
                    / batchSizeToUse)
            } else {
                specifiedBatches = batches
            }
            
            let settings = BenchmarkSettings(
                batches: specifiedBatches ?? defaults.batches,
                batchSize: batchSizeToUse,
                iterations: iterations ?? defaults.iterations,
                warmupBatches: warmupBatches ?? defaults.warmupBatches,
                synthetic: synthetic)

            runBenchmark(
                benchmarkModel,
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
            for (_, benchmarkModel) in benchmarkModels {
                runBenchmark(
                    benchmarkModel,
                    settings: benchmarkModel.defaults(for: .trainingThroughput),
                    variety: .trainingThroughput,
                    style: style)
                runBenchmark(
                    benchmarkModel,
                    settings: benchmarkModel.defaults(for: .inferenceThroughput),
                    variety: .inferenceThroughput,
                    style: style)
            }
        }
    }
}

BenchmarkCommand.main()
