import TensorFlow

let batchSize: Int32 = 4

let (trainingDataset, testDataset) = loadCIFAR10()
let trainingBatches = trainingDataset.batched(Int64(batchSize))
let testBatches = testDataset.batched(Int64(batchSize))

let modeRef = ModeRef()
var net = CIFARNet() // modeRef: modeRef)

// optimizer used in the PyTorch code
let optimizer = SGD<CIFARNet, Float>(learningRate: 0.001, momentum: 0.9)
// optimizer used in the Keras code
// let optimizer = RMSProp<CIFARNet, Float>(learningRate: 0.0001, decay: 1e-6)

for epoch in 1...10 {
    print("Epoch \(epoch), training...")
    modeRef.training = true
    var trainingLossSum: Float = 0
    var trainingBatchCount = 0
    for batch in trainingBatches {
        let gradients = gradient(at: net) {
            (model: CIFARNet) -> Tensor<Float> in
            let thisLoss = loss(
                model: model, images: batch.second, labels: batch.first)
            trainingLossSum += thisLoss.scalarized()
            trainingBatchCount += 1
            return thisLoss
        }
        optimizer.update(&net.allDifferentiableVariables, along: gradients)
    }
    print("  average loss: \(trainingLossSum / Float(trainingBatchCount))")
    print("Epoch \(epoch), evaluating on test set...")
    modeRef.training = false
    var testLossSum: Float = 0
    var testBatchCount = 0
    for batch in testBatches {
        testLossSum += loss(
            model: net, images: batch.second, labels: batch.first).scalarized()
        testBatchCount += 1
    }
    print("  average loss: \(testLossSum / Float(testBatchCount))")
}
