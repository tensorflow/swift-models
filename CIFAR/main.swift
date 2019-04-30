import TensorFlow
import Python
PythonLibrary.useVersion(3)

let batchSize = 100

let cifarDataset = loadCIFAR10()
let testBatches = cifarDataset.test.batched(batchSize)

var model = KerasModel()
let optimizer = RMSProp(for: model, learningRate: 0.0001, decay: 1e-6)

print("Starting training...")
Context.local.learningPhase = .training

for epoch in 1...100 {
    var trainingLossSum: Float = 0
    var trainingBatchCount = 0
    let trainingShuffled = cifarDataset.training.shuffled(
        sampleCount: 50000, randomSeed: Int64(epoch))
    for batch in trainingShuffled.batched(batchSize) {
        let (labels, images) = (batch.label, batch.data)
        let (loss, gradients) = valueWithGradient(at: model) { model -> Tensor<Float> in
            let logits = model(images)
            return softmaxCrossEntropy(logits: logits, labels: labels)
        }
        trainingLossSum += loss.scalarized()
        trainingBatchCount += 1
        optimizer.update(&model.allDifferentiableVariables, along: gradients)
    }

    var testLossSum: Float = 0
    var testBatchCount = 0
    var correctGuessCount = 0
    var totalGuessCount = 0
    for batch in testBatches {
        let (labels, images) = (batch.label, batch.data)
        let logits = model(images)
        testLossSum += softmaxCrossEntropy(logits: logits, labels: labels).scalarized()
        testBatchCount += 1

        let correctPredictions = logits.argmax(squeezingAxis: 1) .== labels
        correctGuessCount = correctGuessCount +
            Int(Tensor<Int32>(correctPredictions).sum().scalarized())
        totalGuessCount = totalGuessCount + batchSize
    }

    let accuracy = Float(correctGuessCount) / Float(totalGuessCount)
    print("""
          [Epoch \(epoch)] \
          Accuracy: \(correctGuessCount)/\(totalGuessCount) (\(accuracy)) \
          Loss: \(testLossSum / Float(testBatchCount))
          """)
}
