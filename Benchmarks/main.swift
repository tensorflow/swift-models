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

func runBenchmark(
    _ name: String,
    withSettings settings: BenchmarkSettings,
    andVariety variety: BenchmarkVariety
) {
    if let benchmarkModel = benchmarkModels[name] {
        var bench: Benchmark
        var benchSettings: BenchmarkSettings
        var varietyName: String
        switch variety {
        case .inferenceThroughput:
            benchSettings =
                settings.withDefaults(benchmarkModel.inferenceDefaults())
            bench = benchmarkModel.inferenceBenchmark(benchSettings)
            varietyName = "inference"
        case .trainingTime:
            benchSettings =
                settings.withDefaults(benchmarkModel.trainingDefaults())
            bench = benchmarkModel.trainingBenchmark(benchSettings)
            varietyName = "training"
        }
        benchmark(
            name: "\(name) (\(varietyName))",
            settings: benchSettings,
            variety: variety,
            benchmark: bench,
            callback: logResults)
    } else {
        print("No registered inference benchmark with a name: \(name)")
        print("Consider running `list` command to see all available benchmarks.")
    }
}

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

let main =
    Group { group in
        group.command(
            "measure-all",
            description: "Run all benchmarks with default settings."
        ) {
            for (name, benchmarkModel) in benchmarkModels {
                runBenchmark(
                    name,
                    withSettings: benchmarkModel.trainingDefaults(),
                    andVariety: .trainingTime)
                runBenchmark(
                    name,
                    withSettings: benchmarkModel.inferenceDefaults(),
                    andVariety: .inferenceThroughput)
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
            description: "Run a single benchmark with provided settings."
        ) { (trainingFlag, inferenceFlag, name, batches, batchSize, iterations, epochs) in
            let settings = BenchmarkSettings(
                batches: batches,
                batchSize: batchSize,
                iterations: iterations,
                epochs: epochs)
            if !trainingFlag && !inferenceFlag {
                print("Must specify either --training xor --inference benchmark variety.")
            } else if trainingFlag && inferenceFlag {
                print("Can't specify both --training and --inference benchmark variety.")
            } else if name == "" {
                print("Must provide a --benchmark to run.")
            } else {
                var variety: BenchmarkVariety
                if trainingFlag {
                    variety = .trainingTime
                } else {
                    assert(inferenceFlag)
                    variety = .inferenceThroughput
                }
                runBenchmark(name, withSettings: settings, andVariety: variety)
            }
        }
        group.command(
            "list-defaults",
            description: "List all available benchmarks and their default settings."
        ) {
            for (name, benchmarkModel) in benchmarkModels {
                print(
                    "--benchmark \(name) --training \(flags(from: benchmarkModel.trainingDefaults()))"
                )
                print(
                    "--benchmark \(name) --inference \(flags(from: benchmarkModel.inferenceDefaults()))"
                )
            }
        }
    }

main.run()
