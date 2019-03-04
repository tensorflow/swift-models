import TensorFlow

let batchSize: Int32 = 4

let (trainingDataset, testDataset) = loadCIFAR10()
let trainingBatches = trainingDataset.batched(Int64(batchSize))
let testBatches = testDataset.batched(Int64(batchSize))

let trainContext = Context(learningPhase: LearningPhase.training)
let testContext = Context(learningPhase: LearningPhase.inference)
var model = CIFARModel()

// optimizer used in the PyTorch code
let optimizer = SGD<CIFARModel, Float>(learningRate: 0.001, momentum: 0.9)
// optimizer used in the Keras code
// let optimizer = RMSProp<CIFARModel, Float>(learningRate: 0.0001, decay: 1e-6)

for epoch in 1...10 {
    print("Epoch \(epoch), training...")
    var trainingLossSum: Float = 0
    var trainingBatchCount = 0
    for batch in trainingBatches {
        let gradients = gradient(at: model) {
            (model: CIFARModel) -> Tensor<Float> in
            let thisLoss = loss(
                model: model, images: batch.second, labels: batch.first, in: trainContext)
            trainingLossSum += thisLoss.scalarized()
            trainingBatchCount += 1
            return thisLoss
        }
        optimizer.update(&model.allDifferentiableVariables, along: gradients)
    }
    print("  average loss: \(trainingLossSum / Float(trainingBatchCount))")
    print("Epoch \(epoch), evaluating on test set...")
    var testLossSum: Float = 0
    var testBatchCount = 0
    for batch in testBatches {
        testLossSum += loss(
        model: model, images: batch.second, labels: batch.first, in: testContext).scalarized()
        testBatchCount += 1
    }
    print("  average loss: \(testLossSum / Float(testBatchCount))")
}
