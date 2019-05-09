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

import TensorFlow
import Python
PythonLibrary.useVersion(3)

let batchSize = 100

let cifarDataset = loadCIFAR10()
let testBatches = cifarDataset.test.batched(batchSize)

// Use the network sized for CIFAR-10
var model = ResNet(kind: .resNet50, imageSize: 32, classCount: 10)

// the classic ImageNet optimizer setting diverges on CIFAR-10
// let optimizer = SGD(for: model, learningRate: 0.1, momentum: 0.9)
let optimizer = SGD(for: model, learningRate: 0.001)

print("Starting training...")
Context.local.learningPhase = .training

for epoch in 1...10 {
    var trainingLossSum: Float = 0
    var trainingBatchCount = 0
    let trainingShuffled = cifarDataset.training.shuffled(
        sampleCount: 50000, randomSeed: Int64(epoch))
    for batch in trainingShuffled.batched(batchSize) {
        let (labels, images) = (batch.label, batch.data)
        let (loss, gradients) = valueWithGradient(at: model) { model -> Tensor<Float> in
            let logits = model(images)
            return softmaxCrossEntropy(logits: logits, labels: labels)
        }
        trainingLossSum += loss.scalarized()
        trainingBatchCount += 1
        optimizer.update(&model.allDifferentiableVariables, along: gradients)
    }
    var testLossSum: Float = 0
    var testBatchCount = 0
    var correctGuessCount = 0
    var totalGuessCount = 0
    for batch in testBatches {
        let (labels, images) = (batch.label, batch.data)
        let logits = model(images)
        testLossSum += softmaxCrossEntropy(logits: logits, labels: labels).scalarized()
        testBatchCount += 1

        let correctPredictions = logits.argmax(squeezingAxis: 1) .== labels
        correctGuessCount = correctGuessCount +
            Int(Tensor<Int32>(correctPredictions).sum().scalarized())
        totalGuessCount = totalGuessCount + batchSize
    }

    let accuracy = Float(correctGuessCount) / Float(totalGuessCount)
    print("""
          [Epoch \(epoch)] \
          Accuracy: \(correctGuessCount)/\(totalGuessCount) (\(accuracy)) \
          Loss: \(testLossSum / Float(testBatchCount))
          """)
}
