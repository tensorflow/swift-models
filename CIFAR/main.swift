import TensorFlow

let batchSize: Int32 = 100

let (trainingDataset, testDataset) = loadCIFAR10()
let testBatches = testDataset.batched(Int64(batchSize))

//var model = ResNet20()
//var model = PyTorchModel()
var model = KerasModel()

// optimizer used in the PyTorch code
// let optimizer = SGD(for: model, learningRate: 0.001, momentum: 0.9, scalarType: Float.self)
// optimizer used in the Keras code
let optimizer = RMSProp(for: model, learningRate: 0.0001, decay: 1e-6, scalarType: Float.self)

print("Starting training...")
for epoch in 1...100 {
    var trainingLossSum: Float = 0
    var trainingBatchCount = 0
    let trainingShuffled = trainingDataset.shuffled(sampleCount: 50000, randomSeed: Int64(epoch))
    for batch in trainingShuffled.batched(Int64(batchSize)) {
        let (labels, images) = (batch.label, batch.data)
        let (loss, gradients) = valueWithGradient(at: model) { model -> Tensor<Float> in
            let logits = model.applied(to: images, in: Context(learningPhase: .training))
            return softmaxCrossEntropy(logits: logits, labels: labels)
        }
        trainingLossSum += loss.scalarized()
        trainingBatchCount += 1
        optimizer.update(&model.allDifferentiableVariables, along: gradients)
    }

    var testLossSum: Float = 0
    var testBatchCount = 0
    var correctGuessCount = 0
    var totalGuessCount: Int32 = 0
    for batch in testBatches {
        let (labels, images) = (batch.label, batch.data)
        let logits = model.inferring(from: images)
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
