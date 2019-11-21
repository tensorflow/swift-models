import Datasets
import ImageClassificationModels

struct LeNetMnist: BenchmarkModel {
    func inferenceDefaults() -> BenchmarkSettings {
        return BenchmarkSettings(batches: 1000, batchSize: 1, iterations: 10, epochs: -1)
    }

    func inferenceBenchmark(_ settings: BenchmarkSettings) -> Benchmark {
        return ImageClassificationInference<LeNet, MNIST>(settings: settings)
    }

    func trainingDefaults() -> BenchmarkSettings {
        return BenchmarkSettings(batches: -1, batchSize: 128, iterations: 10, epochs: 1)
    }

    func trainingBenchmark(_ settings: BenchmarkSettings) -> Benchmark {
        return ImageClassificationTraining<LeNet, MNIST>(settings: settings)
    }
}

let benchmarkModels = [
    "lenet-mnist": LeNetMnist(),
]
