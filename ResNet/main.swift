import TensorFlow

let batchSize = 128

let (trainingDataset, testDataset) = loadCIFAR10()
let testBatches = testDataset.batched(Int64(batchSize))

var model = ResNet50(imageSize: 32, classCount: 10) // Use the network sized for CIFAR-10

// the classic ImageNet optimizer setting diverges on CIFAR-10
// let optimizer = SGD(for: model, learningRate: 0.1, momentum: 0.9, scalarType: Float.self)
let optimizer = SGD(for: model, learningRate: 0.001, scalarType: Float.self)

for epoch in 1...10 {
    print("Epoch \(epoch), training...")
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
    print("  average loss: \(trainingLossSum / Float(trainingBatchCount))")
    print("Epoch \(epoch), evaluating on test set...")
    var testLossSum: Float = 0
    var testBatchCount = 0
    for batch in testBatches {
        let (labels, images) = (batch.label, batch.data)
        let logits = model.inferring(from: images)
        testLossSum += softmaxCrossEntropy(logits: logits, labels: labels).scalarized()
        testBatchCount += 1
    }
    print("  average loss: \(testLossSum / Float(testBatchCount))")
}
