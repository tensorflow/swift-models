// Original source:
// "Big Transfer (BiT): General Visual Representation Learning"
// Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, Neil Houlsby.
// https://arxiv.org/abs/1912.11370

import Datasets
import ImageClassificationModels
import TensorFlow
import Foundation
import PythonKit

let tf = Python.import("tensorflow")
let np  = Python.import("numpy")

// Optional to enable GPU training
// let _ = _ExecutionContext.global
// let device = Device.defaultXLA
let device = Device.default
let modelName = "BiT-M-R50x1"
var knownModels = [String: String]()
let knownDatasetSizes:[String: (Int, Int)] = [
  "cifar10": (32, 32),
  "cifar100": (32, 32),
  "oxford_iiit_pet": (224, 224),
  "oxford_flowers102": (224, 224),
  "imagenet2012": (224, 224),
]

let cifar100TrainingSize = 50000

// Suggested using smaller batch size with no GPU
let batchSize = 128

enum ValueError: Error {
  case invalidInput(String)
}

func getModelUnits(modelName: String) -> BigTransfer.Depth {
  if modelName.contains("R50") {
    return .resNet50
  }
  else if modelName.contains("R101") {
    return .resNet101
  }
  else {
    return .resNet152
  }
}

func getResolution(originalResolution: (Int, Int)) -> (Int, Int) {
  let area = originalResolution.0 * originalResolution.1
  return area < 96*96 ? (160, 128) : (512, 480)
}

func getResolutionFromDataset(dataset: String) throws -> (Int, Int) {
  if let resolution = knownDatasetSizes[dataset] {
    return getResolution(originalResolution: resolution)
  }
  print("Unsupported dataset " + dataset + ". Add your own here :)")
  throw ValueError.invalidInput(dataset)

}

func getMixUp(datasetSize: Int) -> Double {
  return datasetSize < 20000 ? 0.0 : 0.1
}

func getSchedule(datasetSize: Int) -> Array<Int> {
  if datasetSize < 20000{
    return [100, 200, 300, 400, 500]
  }
  else if datasetSize < 500000 {
    return [500, 3000, 6000, 9000, 10000]
  }
  else {
    return [500, 6000, 12000, 18000, 20000]
  }
}


func getLearningRate(step: Int, datasetSize: Int, baseLearningRate: Float = 0.003) -> Float? {
  /* Returns learning-rate for `step` or nil at the end. */
  let supports = getSchedule(datasetSize: datasetSize)
  // Linear warmup
  if step < supports[0] {
    return baseLearningRate * Float(step) / Float(supports[0])
  }
  // End of training
  else if step >= supports.last! {
    return nil
  }
  // Staircase decays by factor of 10
  else {
    var baseLearningRate = baseLearningRate
    for s in supports[1...] {
      if s < step {
        baseLearningRate = baseLearningRate / 10.0
      }
    }
    return baseLearningRate
  }
}

struct Statistics {
  var correctGuessCount = Tensor<Int32>(0, on: Device.default)
  var totalGuessCount = Tensor<Int32>(0, on: Device.default)
  var totalLoss = Tensor<Float>(0, on: Device.default)
  var batches: Int = 0
  var accuracy: Float { 
      Float(correctGuessCount.scalarized()) / Float(totalGuessCount.scalarized()) * 100 
  } 
  var averageLoss: Float { totalLoss.scalarized() / Float(batches) }

  init(on device: Device = Device.default) {
    correctGuessCount = Tensor<Int32>(0, on: device)
    totalGuessCount = Tensor<Int32>(0, on: device)
    totalLoss = Tensor<Float>(0, on: device)
  }

  mutating func update(logits: Tensor<Float>, labels: Tensor<Float>, loss: Tensor<Float>) {
    let correct = logits.argmax(squeezingAxis: 1) .== labels.argmax(squeezingAxis: 1)
    correctGuessCount += Tensor<Int32>(correct).sum()
    totalGuessCount += Int32(labels.shape[0])
    totalLoss += loss
    batches += 1
  }
}

var bitModel = BigTransfer(classCount: 100, depth: getModelUnits(modelName: modelName), modelName: modelName)
let dataset = CIFAR100(batchSize: batchSize, on: Device.default)
var optimizer = SGD(for: bitModel, learningRate: 0.001, momentum: 0.9)
optimizer = SGD(copying: optimizer, to: device)

print("Beginning training...")
var currStep: Int = 1
let lrSupports = getSchedule(datasetSize: cifar100TrainingSize)
let scheduleLength = lrSupports.last!
let stepsPerEpoch = cifar100TrainingSize / batchSize
let epochCount = scheduleLength / stepsPerEpoch
let resizeSize = getResolution(originalResolution: (32, 32))
let mixupAlpha = getMixUp(datasetSize: cifar100TrainingSize)
let beta = np.random.beta(mixupAlpha, mixupAlpha)

for (epoch, batches) in dataset.training.prefix(epochCount).enumerated() {
    let start = Date()
    var trainStats = Statistics(on: device)
    var testStats = Statistics(on: device)
    
    Context.local.learningPhase = .training
    for batch in batches {
      if let newLearningRate = getLearningRate(step: currStep, datasetSize: cifar100TrainingSize, baseLearningRate: 0.003) {
        optimizer.learningRate = newLearningRate
        currStep = currStep + 1
      }
      else {
        continue
      }

      var (eagerImages, eagerLabels) = (batch.data, batch.label)
      let npImages = eagerImages.makeNumpyArray()
      let resized = tf.image.resize(npImages, [resizeSize.0, resizeSize.0])
      let cropped = tf.image.random_crop(resized, [batchSize, resizeSize.1, resizeSize.1, 3])
      let flipped = tf.image.random_flip_left_right(cropped)
      var mixedUp = flipped
      var newLabels = Tensor<Float>(Tensor<Int32>(oneHotAtIndices: eagerLabels, depth: 100))
      if mixupAlpha > 0.0 {
        var npLabels = newLabels.makeNumpyArray()
        mixedUp = beta * mixedUp + (1 - beta) * tf.reverse(mixedUp, axis: [0])
        npLabels = beta * npLabels + (1 - beta) * tf.reverse(npLabels, axis: [0])
        newLabels = Tensor<Float>(numpy: npLabels.numpy())!
      }
      eagerImages = Tensor<Float>(numpy: mixedUp.numpy())!
      let images = Tensor(copying: eagerImages, to: device)
      let labels = Tensor(copying: newLabels, to: device)

      let 𝛁model = TensorFlow.gradient(at: bitModel) { bitModel -> Tensor<Float> in
        let ŷ = bitModel(images)
        let loss = softmaxCrossEntropy(logits: ŷ, probabilities: labels)
        trainStats.update(logits: ŷ, labels: labels, loss: loss)
        return loss
      }

      optimizer.update(&bitModel, along: 𝛁model)
      LazyTensorBarrier()
    }

    Context.local.learningPhase = .inference
    for batch in dataset.validation {
      var (eagerImages, eagerLabels) = (batch.data, batch.label)
      let npImages = eagerImages.makeNumpyArray()
      let resized = tf.image.resize(npImages, [resizeSize.0, resizeSize.0])
      eagerImages = Tensor<Float>(numpy: resized.numpy())!
      let newLabels = Tensor<Float>(Tensor<Int32>(oneHotAtIndices: eagerLabels, depth: 100))
      let images = Tensor(copying: eagerImages, to: device)
      let labels = Tensor(copying: newLabels, to: device)
      let ŷ = bitModel(images)
      let loss = softmaxCrossEntropy(logits: ŷ, probabilities: labels)
      LazyTensorBarrier()
      testStats.update(logits: ŷ, labels: labels, loss: loss)
    }

    print(
      """
      [Epoch \(epoch)] \
      Training Loss: \(String(format: "%.3f", trainStats.averageLoss)), \
      Training Accuracy: \(trainStats.correctGuessCount)/\(trainStats.totalGuessCount) \
      (\(String(format: "%.1f", trainStats.accuracy))%), \
      Test Loss: \(String(format: "%.3f", testStats.averageLoss)), \
      Test Accuracy: \(testStats.correctGuessCount)/\(testStats.totalGuessCount) \
      (\(String(format: "%.1f", testStats.accuracy))%) \
      seconds per epoch: \(String(format: "%.1f", Date().timeIntervalSince(start)))
      """)
}