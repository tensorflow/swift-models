import TensorFlow
import Datasets
import ImageClassificationModels

final class LeNetBenchmark {
    let epochs: Int
    let batchSize: Int
    let dataset: MNIST
    var inferenceModel: LeNet!
    var inferenceImages: Tensor<Float>!
    var batches: Int!
    var inferenceBatches: Int!

    init(epochs: Int, batchSize: Int) {
        self.epochs = epochs
        self.batchSize = batchSize
        self.dataset = MNIST(batchSize: batchSize)
    }

    func performTraining() {
        var model = LeNet()
        let optimizer = SGD(for: model, learningRate: 0.1)

        Context.local.learningPhase = .training
        for _ in 1...epochs {
            for i in 0..<dataset.trainingSize / batchSize {
                let x = dataset.trainingImages.minibatch(at: i, batchSize: batchSize)
                let y = dataset.trainingLabels.minibatch(at: i, batchSize: batchSize)
                let 𝛁model = model.gradient { model -> Tensor<Float> in
                    let ŷ = model(x)
                    return softmaxCrossEntropy(logits: ŷ, labels: y)
                }
                optimizer.update(&model, along: 𝛁model)
            }
        }
    }

    func setupInference(batches: Int, batchSize: Int) {
        inferenceBatches = batches
        inferenceModel = LeNet()
        inferenceImages = Tensor<Float>(
            randomNormal: [batchSize, 28, 28, 1], mean: Tensor<Float>(0.5),
            standardDeviation: Tensor<Float>(0.1), seed: (0xffeffe, 0xfffe))
    }

    func performInference() {
        for _ in 0..<inferenceBatches {
            let _ = inferenceModel(inferenceImages)
        }
    }
}
