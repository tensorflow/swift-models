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
import Datasets
import ImageClassificationModels

let trainingBenchmarks = [
    "lenet-mnist": { ImageClassificationTraining<LeNet, MNIST>(withSettings: $0) },
]

let inferenceBenchmarks = [
    "lenet-mnist": { ImageClassificationInference<LeNet, MNIST>(withSettings: $0) },
]

func runTrainingBenchmark(_ name: String, withSettings settings: BenchmarkSettings) {
    if let makeBenchmark = trainingBenchmarks[name] {
        let bench = makeBenchmark(settings)
        benchmark(
            name: "\(name) (training)",
            settings: settings,
            variety: .trainingTime,
            benchmark: bench,
            callback: logResults)
    } else {
        print("No registered training benchmark with a name: \(name)")
        print("Consider running `list` command to see all available benchmarks.")
    }
}

func runInferenceBenchmark(_ name: String, withSettings settings: BenchmarkSettings) {
    if let makeBenchmark = inferenceBenchmarks[name] {
        let bench = makeBenchmark(settings)
        benchmark(
            name: "\(name) (inference)",
            settings: settings,
            variety: .inferenceThroughput,
            benchmark: bench,
            callback: logResults)
    } else {
        print("No registered inference benchmark with a name: \(name)")
        print("Consider running `list` command to see all available benchmarks.")
    }
}

let main =
    Group {
        group
        ingroup.command(
            "measure",
            Flag("training"),
            Flag("inference"),
            Option("benchmark", default: ""),
            Option("batches", default: 1000),
            Option("batchSize", default: 1),
            Option("iterations", default: 10),
            Option("epochs", default: 1)
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
                if trainingFlag {
                    runTrainingBenchmark(name, withSettings: settings)
                }
                if inferenceFlag {
                    runInferenceBenchmark(name, withSettings: settings)
                }
            }
        }
        group.command("list") {
            print("Registered training benchmarks:")
            for (name, _) in trainingBenchmarks {
                print("  * \(name)")
            }
            print("Registered inference benchmarks:")
            for (name, _) in inferenceBenchmarks {
                print("  * \(name)")
            }
        }
    }

main.run()
