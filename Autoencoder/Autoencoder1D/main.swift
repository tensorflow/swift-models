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

import AutoencoderCallback
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

var trainingLoop = TrainingLoop(
  training: dataset.training.map { $0.map { LabeledData(data: $0.data, label: $0.data) } },
  validation: dataset.validation.map { LabeledData(data: $0.data, label: $0.data) },
  optimizer: optimizer,
  lossFunction: meanSquaredError,
  callbacks: [
    imageSaver(batchSize: batchSize, imageWidth: imageWidth, imageHeight: imageHeight)
  ])

try! trainingLoop.fit(&autoencoder, epochs: epochCount, on: Device.default)
