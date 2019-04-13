import TensorFlow
import Foundation

let imageDirectory = "imagenette-160"
//let imageDirectory = "imagewoof-160"
let inputImageSize = 128

let trainingImageDirectoryURL = URL(fileURLWithPath:"\(imageDirectory)/train")
let trainingImageDataset = try! ImageDataset(imageDirectory: trainingImageDirectoryURL, imageSize: (inputImageSize, inputImageSize))
let validationImageDirectoryURL = URL(fileURLWithPath:"\(imageDirectory)/val")
let validationImageDataset = try! ImageDataset(imageDirectory: validationImageDirectoryURL, imageSize: (inputImageSize, inputImageSize))

let classCount = trainingImageDataset.classes
let totalTrainingImages = trainingImageDataset.imageData.shape[0]
let totalValidationImages = validationImageDataset.imageData.shape[0]

var model = BasicCNNModel()
let optimizer = SGD(for: model, learningRate: 0.001, momentum: 0.9, nesterov: true, scalarType: Float.self)

print("Starting training...")
for epoch in 1...20 {
    var trainingLossSum: Float = 0
    var trainingBatchCount = 0
    let shuffledDataset = trainingImageDataset.combinedDataset.shuffled(sampleCount: Int64(totalTrainingImages), randomSeed: Int64(epoch))

    for batch in shuffledDataset.batched(Int64(42)) {
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
    let totalGuessCount: Int32 = totalValidationImages

    for batch in validationImageDataset.combinedDataset.batched(Int64(50)) {
        let (labels, images) = (batch.label, batch.data)
        let logits = model.inferring(from: images)
        testLossSum += softmaxCrossEntropy(logits: logits, labels: labels).scalarized()
        testBatchCount += 1

        let correctPredictions = logits.argmax(squeezingAxis: 1) .== labels
        correctGuessCount = correctGuessCount +
            Int(Tensor<Int32>(correctPredictions).sum().scalarized())
    }

    let accuracy = Float(correctGuessCount) / Float(totalGuessCount)
    print("""
          [Epoch \(epoch)] \
          Accuracy: \(correctGuessCount)/\(totalGuessCount) (\(accuracy)) \
          Loss: \(testLossSum / Float(testBatchCount))
          """)
}
