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

var gpt = try GPT2()

let sequenceLength = gpt.contextSize
let trainingBatchSize = 2
let validationBatchSize = 2
let numWorkers = 1
// Use default WikiText2 dataset.
let dataset = TextUnsupervised(bpe: gpt.bpe, variant: .wikiText2,
    trainingBatchSize: trainingBatchSize, validationBatchSize: validationBatchSize,
    sequenceLength: sequenceLength)

print("Dataset acquired.")

var optimizer = Adam(for: gpt.model, learningRate: 0.001)

print("Starting training...")

let epochCount = 10
for (epoch, epochBatches) in dataset.training.prefix(epochCount).enumerated() {
    Context.local.learningPhase = .training
    var trainingLossSum: Float = 0
    var trainingBatchCount = 0
    for batch in epochBatches {
        let (documents, labels) = (batch.data, batch.label)
        let (loss, gradients) = valueWithGradient(at: gpt.model) { model -> Tensor<Float> in
            let logits = model(documents)
            let shape = logits.shape
            return softmaxCrossEntropy(
                logits: logits.reshaped(to: [shape[0] * shape[1], shape[2]]),
                labels: labels.reshaped(to: [shape[0] * shape[1]])
            )
        }
        trainingLossSum += loss.scalarized()
        trainingBatchCount += 1
        optimizer.update(&gpt.model, along: gradients)
    }

    Context.local.learningPhase = .inference
    var testLossSum: Float = 0
    var testBatchCount = 0
    var correctGuessCount = 0
    var totalGuessCount = 0
    for batch in dataset.validation {
        let (documents, labels) = (batch.data, batch.label)
        let logits = gpt.model(documents)
        let shape = logits.shape
        testLossSum += softmaxCrossEntropy(
            logits: logits.reshaped(to: [shape[0] * shape[1], shape[2]]),
            labels: labels.reshaped(to: [shape[0] * shape[1]])
        ).scalarized()
        testBatchCount += 1

        let correctPredictions = logits.argmax(squeezingAxis: 2) .== labels
        correctGuessCount =
            correctGuessCount
            + Int(
                Tensor<Int32>(correctPredictions).sum().scalarized())
        totalGuessCount = totalGuessCount + (shape[0] * shape[1])
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
