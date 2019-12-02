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
import Datasets

let epochCount = 12
let batchSize = 128

let dataset = MNIST()
// The LeNet-5 model, equivalent to `LeNet` in `ImageClassificationModels`.
var classifier = Sequential {
    Conv2D<Float>(filterShape: (5, 5, 1, 6), padding: .same, activation: relu)
    AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    Conv2D<Float>(filterShape: (5, 5, 6, 16), activation: relu)
    AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    Flatten<Float>()
    Dense<Float>(inputSize: 400, outputSize: 120, activation: relu)
    Dense<Float>(inputSize: 120, outputSize: 84, activation: relu)
    Dense<Float>(inputSize: 84, outputSize: 10)
}

let optimizer = SGD(for: classifier, learningRate: 0.1)

print("Beginning training...")

struct Statistics {
    var correctGuessCount: Int = 0
    var totalGuessCount: Int = 0
    var totalLoss: Float = 0
    var batches: Int = 0
}

let testBatches = dataset.testDataset.batched(batchSize)

// The training loop.
for epoch in 1...epochCount {
    var trainStats = Statistics()
    var testStats = Statistics()
    let trainingShuffled = dataset.trainingDataset.shuffled(
        sampleCount: dataset.trainingExampleCount, randomSeed: Int64(epoch))

    Context.local.learningPhase = .training
    for batch in trainingShuffled.batched(batchSize) {
        let (labels, images) = (batch.label, batch.data)
        // Compute the gradient with respect to the model.
        let 𝛁model = classifier.gradient { classifier -> Tensor<Float> in
            let ŷ = classifier(images)
            let correctPredictions = ŷ.argmax(squeezingAxis: 1) .== labels
            trainStats.correctGuessCount += Int(
                Tensor<Int32>(correctPredictions).sum().scalarized())
            trainStats.totalGuessCount += batchSize
            let loss = softmaxCrossEntropy(logits: ŷ, labels: labels)
            trainStats.totalLoss += loss.scalarized()
            trainStats.batches += 1
            return loss
        }
        // Update the model's differentiable variables along the gradient vector.
        optimizer.update(&classifier, along: 𝛁model)
    }

    Context.local.learningPhase = .inference
    for batch in testBatches {
        let (labels, images) = (batch.label, batch.data)
        // Compute loss on test set
        let ŷ = classifier(images)
        let correctPredictions = ŷ.argmax(squeezingAxis: 1) .== labels
        testStats.correctGuessCount += Int(Tensor<Int32>(correctPredictions).sum().scalarized())
        testStats.totalGuessCount += batchSize
        let loss = softmaxCrossEntropy(logits: ŷ, labels: labels)
        testStats.totalLoss += loss.scalarized()
        testStats.batches += 1
    }

    let trainAccuracy = Float(trainStats.correctGuessCount) / Float(trainStats.totalGuessCount)
    let testAccuracy = Float(testStats.correctGuessCount) / Float(testStats.totalGuessCount)
    print(
        """
        [Epoch \(epoch)] \
        Training Loss: \(trainStats.totalLoss / Float(trainStats.batches)), \
        Training Accuracy: \(trainStats.correctGuessCount)/\(trainStats.totalGuessCount) \
        (\(trainAccuracy)), \
        Test Loss: \(testStats.totalLoss / Float(testStats.batches)), \
        Test Accuracy: \(testStats.correctGuessCount)/\(testStats.totalGuessCount) \
        (\(testAccuracy))
        """)
}
