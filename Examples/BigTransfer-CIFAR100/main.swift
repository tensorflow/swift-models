// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Original source:
// "Big Transfer (BiT): General Visual Representation Learning"
// Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, Neil Houlsby.
// https://arxiv.org/abs/1912.11370

import Datasets
import ImageClassificationModels
import TensorFlow
import Foundation
import PythonKit

// let tf = Python.import("tensorflow")
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
let batchSize = 128

/// This error indicates that BiT-Hyperrule cannot find the name of the dataset in the
/// knownDatasetSizes dictionary
enum DatasetNotFoundError: Error {
  case invalidInput(String)
}

/// Return relevent ResNet enumerated type based on weights loaded
///
/// - Parameters:
///   - modelName: the name of the model pulled from the big transfer repository
///                to grab the enumerated type for
/// - Returns: ResNet enumerated type for BigTransfer model
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

/// Get updated image resolution based on the specifications in BiT-Hyperrule
///
/// - Parameters:
///   - originalResolution: the source resolution for the current image dataset
/// - Returns: new resolution for images based on BiT-Hyperrule
func getResolution(originalResolution: (Int, Int)) -> (Int, Int) {
  let area = originalResolution.0 * originalResolution.1
  return area < 96*96 ? (160, 128) : (512, 480)
}

/// Get the source resolution for the current image dataset from the knownDatasetSizes dictionary
///
/// - Parameters:
///   - datasetName: name of the current dataset you are using
/// - Returns: new resolution for specified dataset
/// - Throws:
///   - DatasetNotFoundError: will throw an error if the dataset cannot be found in knownDatasetSizes dictionary
func getResolutionFromDataset(datasetName: String) throws -> (Int, Int) {
  if let resolution = knownDatasetSizes[datasetName] {
    return getResolution(originalResolution: resolution)
  }
  print("Unsupported dataset " + datasetName + ". Add your own here :)")
  throw DatasetNotFoundError.invalidInput(datasetName)

}

/// Get training mixup parameters based on Bit-Hyperrule specification for dataset sizes
///
/// - Parameters:
///   - datasetSize: number of images in the current dataset
/// - Returns: mixup alpha based on number of images
func getMixUp(datasetSize: Int) -> Double {
  return datasetSize < 20000 ? 0.0 : 0.1
}

/// Get the learning rate schedule based on the dataset size
///
/// - Parameters:
///   - datasetSize: number of images in the current dataset
/// - Returns: learning rate schedule based on the current dataset
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

/// Get learning rate at the current step given the dataset size and base learning rate
///
/// - Parameters:
///   - step: current training step
///   - datasetSize: number of images in the dataset
///   - baseLearningRate: starting learning rate to modify
/// - Returns: learning rate at the current step in training
func getLearningRate(step: Int, datasetSize: Int, baseLearningRate: Float = 0.003) -> Float? {
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

/// Stores the training statistics for the BigTransfer training process which are different than usual
/// because the mixedup labels must be accounted for while running training statistics.
struct BigTransferTrainingStatistics {
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

let classCount = 100
var bitModel = BigTransfer(classCount: classCount, depth: getModelUnits(modelName: modelName), modelName: modelName)
let dataset = CIFAR100(batchSize: batchSize, on: Device.default)
var optimizer = SGD(for: bitModel, learningRate: 0.003, momentum: 0.9)
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
    var trainStats = BigTransferTrainingStatistics(on: device)
    var testStats = BigTransferTrainingStatistics(on: device)
    
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
      let resized = resize(images: eagerImages, size: (resizeSize.0, resizeSize.1))
      // Future work to change these calls from Python TensorFlow to Swift for Tensorflow
      // let cropped = tf.image.random_crop(resized, [batchSize, resizeSize.1, resizeSize.1, 3])
      // let flipped = tf.image.random_flip_left_right(cropped)
      // var mixedUp = flipped
      var newLabels = Tensor<Float>(Tensor<Int32>(oneHotAtIndices: eagerLabels, depth: classCount))
      //if mixupAlpha > 0.0 {
      //  var npLabels = newLabels.makeNumpyArray()
      //  mixedUp = beta * mixedUp + (1 - beta) * tf.reverse(mixedUp, axis: [0])
      //  npLabels = beta * npLabels + (1 - beta) * tf.reverse(npLabels, axis: [0])
      //  newLabels = Tensor<Float>(numpy: npLabels.numpy())!
      //}
      // eagerImages = Tensor<Float>(numpy: mixedUp.numpy())!
      // let images = Tensor(copying: eagerImages, to: device)
      let images = Tensor(copying: resized, to: device)
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
      let resized = resize(images: eagerImages, size: (resizeSize.0, resizeSize.1))
      let newLabels = Tensor<Float>(Tensor<Int32>(oneHotAtIndices: eagerLabels, depth: classCount))
      let images = Tensor(copying: resized, to: device)
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