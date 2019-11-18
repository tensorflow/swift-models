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

import ImageClassificationModels
import Datasets
import Commander

let trainingBenchmarks = [
	"lenet-mnist": { (settings: BenchmarkSettings) in
        let leNetTrainingBenchmark = ImageClassificationTraining<LeNet, MNIST>(
            epochs: settings.epochs, batchSize: settings.batchSize)
        benchmark(
            name: "LeNet-MNIST (training)",
            iterations: 10, variety: .trainingTime, operation: leNetTrainingBenchmark.train,
            callback: logResults)
    }
]

let inferenceBenchmarks = [
    "lenet-mnist": { (settings: BenchmarkSettings) in
        let leNetInferenceBenchmark = ImageClassificationInference<LeNet, MNIST>(
            batches: settings.batches, batchSize: settings.batchSize)
        benchmark(
            name: "LeNet-MNIST (inference)",
            iterations: settings.iterations,
            variety: .inferenceThroughput(batches: settings.batches, batchSize: settings.batchSize),
            operation: leNetInferenceBenchmark.performInference,
            callback: logResults)
    }
]

let main = Group { group in
    group.command(
        "inference",
        Argument<String>("name", description: "Benchmark name."),
        Option("batches", default: 1000),
        Option("batchSize", default: 1),
        Option("iterations", default: 10),
        Option("epochs", default: 1)
    ) { (name, batches, batchSize, iterations, epochs) in
        let settings = BenchmarkSettings(
            batches: batches,
            batchSize: batchSize,
            iterations: iterations,
            epochs: epochs)
        if let runner = inferenceBenchmarks[name] {
            runner(settings)
        } else {
            print("No registered inference benchmark with a name: \(name)")
            print("Consider running `list` command to see all available benchmarks.")
        }
    }
    group.command(
        "training",
        Argument<String>("name", description: "Benchmark name."),
        Option("batches", default: 1000),
        Option("batchSize", default: 1),
        Option("iterations", default: 10),
        Option("epochs", default: 1)
    ) { (name, batches, batchSize, iterations, epochs) in
        let settings = BenchmarkSettings(
            batches: batches,
            batchSize: batchSize,
            iterations: iterations,
            epochs: epochs)
        if let runner = trainingBenchmarks[name] {
            runner(settings)
        } else {
            print("No registered training benchmark with a name: \(name)")
            print("Consider running `list` command to see all available benchmarks.")
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
