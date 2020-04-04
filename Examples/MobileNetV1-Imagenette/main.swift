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

let imagenette = Imagenette(
    batchSize: 64,
    inputSize: .resized320,
    outputSize: 224
)

var model = MobileNetV1(classCount: 10)

let optimizer = SGD(for: model, learningRate: 0.02, momentum: 0.9)

print("Starting training...")

for epoch in 1...10 {
    Context.local.learningPhase = .training
    var trainingLossSum: Float = 0
    var trainingBatchCount = 0
    for batch in imagenette.training.sequenced() {
        let (images, labels) = (batch.first, batch.second)
        let (loss, gradients) = valueWithGradient(at: model) { model -> Tensor<Float> in
            let logits = model(images)
            return softmaxCrossEntropy(logits: logits, labels: labels)
        }
        trainingLossSum += loss.scalarized()
        trainingBatchCount += 1
        optimizer.update(&model, along: gradients)
    }

    Context.local.learningPhase = .inference
    var testLossSum: Float = 0
    var testBatchCount = 0
    var correctGuessCount = 0
    var totalGuessCount = 0
    for batch in imagenette.test.sequenced() {
        let (images, labels) = (batch.first, batch.second)
        let logits = model(images)
        testLossSum += softmaxCrossEntropy(logits: logits, labels: labels).scalarized()
        testBatchCount += 1

        let correctPredictions = logits.argmax(squeezingAxis: 1) .== labels
        correctGuessCount = correctGuessCount
            + Int(
                Tensor<Int32>(correctPredictions).sum().scalarized())
        totalGuessCount = totalGuessCount + batch.first.shape[0]
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
