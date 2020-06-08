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
import ImageClassificationModels
import TensorFlow
import TrainingLoop

// TODO: Replace this when macOS does not segfault on use of X10 here.
#if os(macOS)
  let device = Device.defaultTFEager
#else
  let device = Device.defaultXLA
#endif

let dataset = CIFAR10(batchSize: 10, on: device)

var model = ResNet(classCount: 10, depth: .resNet56, downsamplingInFirstStage: false)
model.move(to: device)

var optimizer = SGD(for: model, learningRate: 0.001)
optimizer = SGD(copying: optimizer, to: device)

let trainingProgress = TrainingProgress()

var trainingLoop = TrainingLoop(
  training: dataset.training,
  validation: dataset.validation,
  model: model,
  optimizer: optimizer,
  lossFunction: softmaxCrossEntropy,
  callbacks: [trainingProgress.update])

try! trainingLoop.fit(epochs: 10)
