import TensorFlow
import Foundation

let imageDirectory = "imagenette-160"
let inputImageSize = 160
let internalImageSize = 128

print("Building dataset...")
let trainingImageDirectoryURL = URL(fileURLWithPath:"\(imageDirectory)/train")
let trainingImageDataset = try! ImageDataset(imageDirectory: trainingImageDirectoryURL,
    imageSize: (inputImageSize, inputImageSize))
let validationImageDirectoryURL = URL(fileURLWithPath:"\(imageDirectory)/val")
let validationImageDataset = try! ImageDataset(imageDirectory: validationImageDirectoryURL,
    imageSize: (inputImageSize, inputImageSize))

let classCount = trainingImageDataset.classes
let totalTrainingImages = trainingImageDataset.imageData.shape[0]
let totalValidationImages = validationImageDataset.imageData.shape[0]

var model = BasicCNNModel()
let optimizer = SGD(for: model, learningRate: 0.001, momentum: 0.9, nesterov: true,
    scalarType: Float.self)

print("Starting training...")
for epoch in 1...80 {
    if epoch > 70 { optimizer.learningRate = 0.0001 }

    var trainingLossSum: Float = 0
    var trainingBatchCount = 0
    let shuffledDataset = trainingImageDataset.combinedDataset.shuffled(
        sampleCount: totalTrainingImages, randomSeed: Int64(epoch))
    let batchSize: Int = 42

    for batch in shuffledDataset.batched(batchSize) {
        let (labels, images) = (batch.label, batch.data)

        var boxList: [Float] = []
        var boxIndiciesList: [Int32] = []
        for i in 1...batchSize {
            let max = inputImageSize - internalImageSize
            let xOffset = Float(Int.random(in: 0..<max)) / Float(inputImageSize)
            let yOffset = Float(Int.random(in: 0..<max)) / Float(inputImageSize)
            let offset = Float(internalImageSize) / Float(inputImageSize)

            if Bool.random() {
                boxList.append(contentsOf:[yOffset, xOffset, yOffset + offset, xOffset + offset])
            } else { // using X2 > X1 to add random flips here
                boxList.append(contentsOf:[yOffset, xOffset + offset, yOffset + offset, xOffset])
            }
            boxIndiciesList.append(Int32(i-1))
        }
        let boxesWrapped = Tensor<Float>(shape:[batchSize, 4], scalars: boxList)
        let boxIndicies = Tensor<Int32>(boxIndiciesList)

        let randomlyCroppedImages = Raw.cropAndResize(image: images, boxes: boxesWrapped,
            boxInd: boxIndicies,
            cropSize:Tensor<Int32>([Int32(internalImageSize), Int32(internalImageSize)]))

        let (loss, gradients) = valueWithGradient(at: model) { model -> Tensor<Float> in
            let logits = model.applied(to: randomlyCroppedImages)
            return softmaxCrossEntropy(logits: logits, labels: labels)
        }
        trainingLossSum += loss.scalarized()
        trainingBatchCount += 1
        optimizer.update(&model.allDifferentiableVariables, along: gradients)
    }

    var testLossSum: Float = 0
    var testBatchCount = 0
    var correctGuessCount = 0
    let totalGuessCount = totalValidationImages
    let testBatchSize = 50

    for batch in validationImageDataset.combinedDataset.batched(testBatchSize) {
        let (labels, images) = (batch.label, batch.data)

        var boxList: [Float] = []
        var boxIndiciesList: [Int32] = []
        for i in 1...testBatchSize {
            let maxX = Float(inputImageSize)
            let maxY = Float(inputImageSize)

            let xPrime = (maxX - Float(internalImageSize)) / 2.0
            let yPrime = (maxY - Float(internalImageSize)) / 2.0

            let xOne = xPrime / maxX
            let yOne = yPrime / maxY
            let xTwo = (xPrime + Float(internalImageSize)) / maxX
            let yTwo = (yPrime + Float(internalImageSize)) / maxY

            boxList.append(contentsOf:[yOne, xOne, yTwo, xTwo])
            boxIndiciesList.append(Int32(i-1))
        }
        let boxesWrapped = Tensor<Float>(shape:[testBatchSize, 4], scalars: boxList)
        let boxIndicies = Tensor<Int32>(boxIndiciesList)

        let centerCroppedImages = Raw.cropAndResize(image: images, boxes: boxesWrapped,
            boxInd: boxIndicies,
            cropSize:Tensor<Int32>([Int32(internalImageSize), Int32(internalImageSize)]))

        let logits = model.inferring(from: centerCroppedImages)
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
