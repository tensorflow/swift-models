import TensorFlow

let batchSize: Int32 = 32

let (trainingDataset, testDataset) = loadCIFAR10()
let testBatches = testDataset.batched(Int64(batchSize))

var model = KerasModel() // or PyTorchModel()

// optimizer used in the PyTorch code
// let optimizer = SGD(for: model, learningRate: 0.001, momentum: 0.9, scalarType: Float.self)
// optimizer used in the Keras code
let optimizer = RMSProp(for: model, learningRate: 0.0001, decay: 1e-6, scalarType: Float.self)

for epoch in 1...10 {
    print("Epoch \(epoch), training...")
    var trainingLossSum: Float = 0
    var trainingBatchCount = 0
    let trainingShuffled = trainingDataset.shuffled(sampleCount: 50000, randomSeed: Int64(epoch))
    for batch in trainingShuffled.batched(Int64(batchSize)) {
        let (labels, images) = (batch.first, batch.second)
        let gradients = gradient(at: model) { model -> Tensor<Float> in
            let logits = model.applied(to: images, in: Context(learningPhase: .training))
            let loss = softmaxCrossEntropy(logits: logits, labels: labels)
            trainingLossSum += loss.scalarized()
            trainingBatchCount += 1
            return loss
        }
        optimizer.update(&model.allDifferentiableVariables, along: gradients)
    }
    print("  average loss: \(trainingLossSum / Float(trainingBatchCount))")
    print("Epoch \(epoch), evaluating on test set...")
    var testLossSum: Float = 0
    var testBatchCount = 0
    for batch in testBatches {
        let (labels, images) = (batch.first, batch.second)
        let logits = model.inferring(from: images)
        testLossSum += softmaxCrossEntropy(logits: logits, labels: labels).scalarized()
        testBatchCount += 1
    }
    print("  average loss: \(testLossSum / Float(testBatchCount))")
}
