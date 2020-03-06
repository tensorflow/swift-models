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

import Batcher
import Datasets
import TensorFlow
import TextModels

var gpt = try GPT2()
var model = gpt.model

let dataset = WikiText2(bpe: gpt.bpe)
let trainingBatcher = Batcher(on: dataset.trainingDataset, batchSize: 1)
let validationBatcher = Batcher(on: dataset.validationDataset, batchSize: 1)

print("Dataset acquired.")

var optimizer = Adam(for: model, learningRate: 0.02)

print("Starting training...")

for epoch in 1...10 {
    Context.local.learningPhase = .training
    var trainingLossSum: Float = 0
    var trainingBatchCount = 0
    for batch in trainingBatcher.sequenced() {
        let (documents, labels) = (batch.first, batch.second)
        let (loss, gradients) = valueWithGradient(at: model) { model -> Tensor<Float> in
            let logits = model(documents)
            let shape = logits.shape
            return softmaxCrossEntropy(logits:
                logits.reshaped(to: [shape[0] * shape[1], shape[2]]),
                labels: labels.reshaped(to: [shape[0] * shape[1]])
            )
        }
        trainingLossSum += loss.scalarized()
        trainingBatchCount += 1
        optimizer.update(&model, along: gradients)
        print("loss: \(loss.scalarized())")
    }

    Context.local.learningPhase = .inference
    var testLossSum: Float = 0
    var testBatchCount = 0
    var correctGuessCount = 0
    var totalGuessCount = 0
    for batch in validationBatcher.sequenced() {
        let (labels, documents) = (batch.first, batch.second)
        let logits = model(documents)
        let shape = logits.shape
        testLossSum += softmaxCrossEntropy(logits:
                logits.reshaped(to: [shape[0] * shape[1], shape[2]]),
                labels: labels.reshaped(to: [shape[0] * shape[1]])
            ).scalarized()
        testBatchCount += 1

        let correctPredictions = logits.argmax(squeezingAxis: 1) .== labels
        correctGuessCount = correctGuessCount
            + Int(
                Tensor<Int32>(correctPredictions).sum().scalarized())
        totalGuessCount = totalGuessCount + documents.shape[0]
    }

    let accuracy = Float(correctGuessCount) / Float(totalGuessCount)
    print(
        """
        [Epoch \(epoch)] \
        Accuracy: \(correctGuessCount)/\(totalGuessCount) (\(accuracy)) \
        Loss: \(testLossSum / Float(testBatchCount))
        """
    )
}

/*
for _ in 0..<100 {
    do {
      try print(gpt.generate(), terminator: "")
    } catch {
      continue
    }
}
print()
*/
