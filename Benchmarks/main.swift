// LeNet-MNIST
let leNetBenchmark = LeNetBenchmark(epochs: 1, batchSize: 128)
benchmarkTraining(
    iterations: 10, operation: leNetBenchmark.performTraining,
    callback: logResults(name: "LeNet-MNIST (training)"))
benchmarkInference(
    iterations: 10, batches: 1000, batchSize: 1, setup: leNetBenchmark.setupInference,
    operation: leNetBenchmark.performInference,
    callback: logResults(name: "LeNet-MNIST (inference)"))
