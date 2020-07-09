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
import TensorFlow
import TrainingLoop
import LayerInit

let epochCount = 12
let batchSize = 128

// Until https://github.com/tensorflow/swift-apis/issues/993 is fixed, default to the eager-mode
// device on macOS instead of X10.
#if os(macOS)
  let device = Device.defaultTFEager
#else
  let device = Device.defaultXLA
#endif

let dataset = MNIST(batchSize: batchSize, on: device)

var classifier =
  input(shape: [28, 28, 1])
    .conv2D(filterShape: (5, 5), outputChannels: 6, padding: .same, activation: relu)
    .avgPool2D(poolSize: (2, 2), strides: (2, 2))
    .conv2D(filterShape: (5, 5), outputChannels: 16, activation: relu)
    .avgPool2D(poolSize: (2, 2), strides: (2, 2))
    .flatten()
    .dense(outputSize: 120, activation: relu)
    .dense(outputSize: 84, activation: relu)
    .dense(outputSize: 10)
    .build()

var optimizer = SGD(for: classifier, learningRate: 0.1)

let trainingProgress = TrainingProgress()
var trainingLoop = TrainingLoop(
  training: dataset.training,
  validation: dataset.validation,
  optimizer: optimizer,
  lossFunction: softmaxCrossEntropy,
  callbacks: [trainingProgress.update])

try! trainingLoop.fit(&classifier, epochs: epochCount, on: device)
