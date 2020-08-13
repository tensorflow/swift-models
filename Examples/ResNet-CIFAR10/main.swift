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
import LayerInit

// Until https://github.com/tensorflow/swift-apis/issues/993 is fixed, default to the eager-mode
// device on macOS instead of X10.
// #if os(macOS)
  let device = Device.defaultTFEager
// #else
//   let device = Device.defaultXLA
// #endif

let dataset = CIFAR10(batchSize: 10, on: device)
var model = ResNet(classCount: 10, depth: .resNet56, downsamplingInFirstStage: false)
var optimizer = SGD(for: model, learningRate: 0.001)

let trainingProgress = TrainingProgress()
var trainingLoop = TrainingLoop(
  training: dataset.training.map({ $0.prefix(32) }),
  validation: dataset.validation,
  optimizer: optimizer,
  lossFunction: softmaxCrossEntropy,
  callbacks: [trainingProgress.update])

// print the weights of the final dense layer
print(model.underlying[model.underlyingStruct.output])

try! trainingLoop.fit(&model, epochs: 1, on: device)

print(model.underlying[model.underlyingStruct.output])

// scratchpad for weight sharing (TODO: clean up build)
if (false) {
  let inputNode = InputTracingLayer(shape: [1])
  let firstDense = inputNode.dense(outputSize: 1, useBias: false)
  let secondDense = firstDense.dense(outputSize: 1, useBias: false).sharingWeights(with: firstDense)
  var builtModel = secondDense.build()

  let sharingOptimizer = SGD(for: builtModel, learningRate: 0.01)

  let x: Tensor<Float> = [[0], [1], [2], [3]]
  let y: Tensor<Float> = [0, 9, 18, 27]

  for _ in 0...10 {
      let 𝛁model = gradient(at: builtModel) { classifier -> Tensor<Float> in
          let ŷ = classifier(x).withDerivative { print("∂L/∂ŷ =", $0) }
          let loss = (ŷ - y).squared().mean()
          print("Loss: \(loss)")
          return loss
      }
      sharingOptimizer.update(&builtModel, along: 𝛁model)

      print(builtModel[firstDense])
      print(builtModel[secondDense])
  }
}
