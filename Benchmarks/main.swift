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

import Commander
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

let main =
    Group { group in
        group.command(
            "measure-all",
            Flag("json", description: "Output json instead of plain text."),
            description: "Run all benchmarks with default settings."
        ) { useJSON in
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
        group.command(
            "measure",
            Flag("training", description: "Run training benchmark."),
            Flag("inference", description: "Run inference benchmark."),
            Option("benchmark", default: "", description: "Name of the benchmark to run."),
            Option("batches", default: -1, description: "Number of batches."),
            Option("batchSize", default: -1, description: "Size of a single batch."),
            Option("iterations", default: -1, description: "Number of benchmark iterations."),
            Option("epochs", default: -1, description: "Number of training epochs."),
            Flag("json", description: "Output json instead of plain text."),
            description: "Run a single benchmark with provided settings."
        ) { (trainingFlag, inferenceFlag, name, batches, batchSize, iterations, epochs, useJSON) in
            let style: PrintingStyle = useJSON ? .json : .plainText
            let settings = BenchmarkSettings(
                batches: batches,
                batchSize: batchSize,
                iterations: iterations,
                epochs: epochs)
            if !trainingFlag && !inferenceFlag {
                printError("Must specify either --training xor --inference benchmark variety.")
            } else if trainingFlag && inferenceFlag {
                printError("Can't specify both --training and --inference benchmark variety.")
            } else if name == "" {
                printError("Must provide a --benchmark to run.")
            } else {
                var variety: BenchmarkVariety
                if trainingFlag {
                    variety = .trainingTime
                } else {
                    assert(inferenceFlag)
                    variety = .inferenceThroughput
                }
                runBenchmark(
                    named: name,
                    settings: settings,
                    variety: variety,
                    style: style
                )
            }
        }
        group.command(
            "list-defaults",
            Flag("json"),
            description: "List all available benchmarks and their default settings."
        ) { useJSON in
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

main.run()
