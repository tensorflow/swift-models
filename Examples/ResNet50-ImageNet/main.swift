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
import TensorBoard
import TensorFlow
import TrainingLoop

// XLA mode can't load Imagenet, need to use eager mode to limit memory use
let device = Device.defaultTFEager
let dataset = ImageNet(batchSize: 32, outputSize: 224, on: device)
var model = ResNet(classCount: 1000, depth: .resNet50)

// 0.1 for 30, .01 for 30, .001 for 30
let optimizer = SGD(for: model, learningRate: 0.1, momentum: 0.9)
public func scheduleLearningRate<L: TrainingLoopProtocol>(
  _ loop: inout L, event: TrainingLoopEvent
) throws where L.Opt.Scalar == Float {
  if event == .epochStart {
    guard let epoch = loop.epochIndex else  { return }
    if epoch > 30 { loop.optimizer.learningRate = 0.01 }
    if epoch > 60 { loop.optimizer.learningRate = 0.001 }
  }
}

var trainingLoop = TrainingLoop(
  training: dataset.training,
  validation: dataset.validation,
  optimizer: optimizer,
  lossFunction: softmaxCrossEntropy,
  metrics: [.accuracy],
  callbacks: [scheduleLearningRate, tensorBoardStatisticsLogger()])

try! trainingLoop.fit(&model, epochs: 90, on: device)
