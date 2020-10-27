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

import Datasets
import TensorFlow
import TextModels
import TrainingLoop

// Avoid the eager mode runtime from taking all memory 
// and leaving none to X10 when run on the GPU.
_ = _ExecutionContext.global
// Until https://github.com/tensorflow/swift-apis/issues/993 is fixed, default to the eager-mode
// device on macOS instead of X10.
#if os(macOS)
  let device = Device.defaultTFEager
#else
  let device = Device.defaultXLA
#endif

var gpt = try GPT2()

let sequenceLength = gpt.contextSize
let trainingBatchSize = 2
let validationBatchSize = 2
let dataset = TextUnsupervised(bpe: gpt.bpe, variant: .wikiText2,
    trainingBatchSize: trainingBatchSize, validationBatchSize: validationBatchSize,
    sequenceLength: sequenceLength, on: device)
print("Dataset acquired.")

/// Reshape the `logits` and `labels` to required shape before calling
/// standard softmaxCrossEntropy API.
///
/// - Note: This can potentially be added to standard softmaxCrossEntropy API.
@differentiable
public func softmaxCrossEntropyReshaped<Scalar>(logits: Tensor<Scalar>, labels: Tensor<Int32>) -> Tensor<
  Scalar
> where Scalar: TensorFlowFloatingPoint {
  return softmaxCrossEntropy(
  	logits: logits.reshaped(to: [logits.shape.dropLast().reduce(1, *), logits.shape.last!]), 
  	labels: labels.reshaped(to: [labels.shape.reduce(1, *)]), 
  	reduction: _mean)
}

var trainingLoop: TrainingLoop = TrainingLoop(
  training: dataset.training,
  validation: dataset.validation,
  optimizer: Adam(for: gpt.model, learningRate: 0.001),
  lossFunction: softmaxCrossEntropyReshaped,
  metrics: [.accuracy],
  callbacks: [tensorBoardStatsWriter()])

print("Starting training...")
try! trainingLoop.fit(&gpt.model, epochs: 10, on: device)
