// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

import Datasets
import Foundation
import ModelSupport
import TensorFlow
import TrainingLoop

let epochCount = 10
let batchSize = 100
let imageHeight = 28
let imageWidth = 28

let dataset = FashionMNIST(
  batchSize: batchSize, device: Device.default,
  entropy: SystemRandomNumberGenerator(), flattening: true)

// An autoencoder.
var autoencoder = Sequential {
  // The encoder.
  Dense<Float>(inputSize: imageHeight * imageWidth, outputSize: 128, activation: relu)
  Dense<Float>(inputSize: 128, outputSize: 64, activation: relu)
  Dense<Float>(inputSize: 64, outputSize: 12, activation: relu)
  Dense<Float>(inputSize: 12, outputSize: 3, activation: relu)
  // The decoder.
  Dense<Float>(inputSize: 3, outputSize: 12, activation: relu)
  Dense<Float>(inputSize: 12, outputSize: 64, activation: relu)
  Dense<Float>(inputSize: 64, outputSize: 128, activation: relu)
  Dense<Float>(inputSize: 128, outputSize: imageHeight * imageWidth, activation: tanh)
}

let optimizer = RMSProp(for: autoencoder)

/// Saves a validation input and an output image once per epoch;
/// it's ensured that each epoch will save different images as long as 
/// count of epochs is less or equal than count of images.
/// 
/// It's defined as a callback registered into TrainingLoop.
func saveImage<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
  if event != .inferencePredictionEnd { return }

  guard let batchIndex = loop.batchIndex,
    let batchCount = loop.batchCount,
    let epochIndex = loop.epochIndex,
    let epochCount = loop.epochCount,
    let input = loop.lastStepInput,
    let output = loop.lastStepOutput
  else {
    return
  }

  let imageCount = batchCount * batchSize
  let selectedImageGlobalIndex = epochIndex * (imageCount / epochCount)
  let selectedBatchIndex = selectedImageGlobalIndex / batchSize

  if batchIndex != selectedBatchIndex { return }

  let outputFolder = "./output/"
  let selectedImageBatchLocalIndex = selectedImageGlobalIndex % batchSize
  let inputExample =
    (input as! Tensor<Float>)[selectedImageBatchLocalIndex..<selectedImageBatchLocalIndex+1]
    .normalizedToGrayscale().reshaped(to: [imageWidth, imageHeight, 1])
  try inputExample.saveImage(
    directory: outputFolder, name: "epoch-\(epochIndex + 1)-of-\(epochCount)-input", format: .png)
  let outputExample =
    (output as! Tensor<Float>)[selectedImageBatchLocalIndex..<selectedImageBatchLocalIndex+1]
    .normalizedToGrayscale().reshaped(to: [imageWidth, imageHeight, 1])
  try outputExample.saveImage(
    directory: outputFolder, name: "epoch-\(epochIndex + 1)-of-\(epochCount)-output", format: .png)
}

var trainingLoop = TrainingLoop(
  training: dataset.training.map { $0.map { LabeledData(data: $0.data, label: $0.data) } },
  validation: dataset.validation.map { LabeledData(data: $0.data, label: $0.data) },
  optimizer: optimizer,
  lossFunction: meanSquaredError,
  callbacks: [saveImage])

try! trainingLoop.fit(&autoencoder, epochs: epochCount, on: Device.default)
