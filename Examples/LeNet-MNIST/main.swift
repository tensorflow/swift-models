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

// Until https://github.com/tensorflow/swift-models/issues/588 is fixed, default to the eager-mode
// device on macOS instead of X10.
#if os(macOS)
  let device = Device.defaultTFEager
#else
  let device = Device.defaultXLA
#endif

let dataset = MNIST(batchSize: batchSize)
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
classifier.move(to: device)

var optimizer = SGD(for: classifier, learningRate: 0.1)
optimizer = SGD(copying: optimizer, to: device)

print("Beginning training...")

struct Statistics {
    var correctGuessCount = Tensor<Int32>(0, on: Device.default)
    var totalGuessCount = Tensor<Int32>(0, on: Device.default)
    var totalLoss = Tensor<Float>(0, on: Device.default)
    var batches: Int = 0

    var accuracy: Float {
        Float(correctGuessCount.scalarized()) / Float(totalGuessCount.scalarized()) * 100
    }

    var averageLoss: Float {
        totalLoss.scalarized() / Float(batches)
    }

    init(on device: Device = Device.default) {
        correctGuessCount = Tensor<Int32>(0, on: device)
        totalGuessCount = Tensor<Int32>(0, on: device)
        totalLoss = Tensor<Float>(0, on: device)
    }

    mutating func update(logits:  Tensor<Float>, labels: Tensor<Int32>, loss: Tensor<Float>) {
        let correct = logits.argmax(squeezingAxis: 1) .== labels
        correctGuessCount += Tensor<Int32>(correct).sum()
        totalGuessCount += Int32(labels.shape[0])
        totalLoss += loss
        batches += 1
    }
}

// The training loop.
for (epoch, epochBatches) in dataset.training.prefix(epochCount).enumerated() {
    var trainStats = Statistics(on: device)
    var testStats = Statistics(on: device)
    
    Context.local.learningPhase = .training
    for batch in epochBatches {
        let (eagerImages, eagerLabels) = (batch.data, batch.label)
        let images = Tensor(copying: eagerImages, to: device)
        let labels = Tensor(copying: eagerLabels, to: device)
        // Compute the gradient with respect to the model.
        let 𝛁model = TensorFlow.gradient(at: classifier) { classifier -> Tensor<Float> in
            let ŷ = classifier(images)
            let loss = softmaxCrossEntropy(logits: ŷ, labels: labels)
            trainStats.update(logits: ŷ, labels: labels, loss: loss)
            return loss
        }
        // Update the model's differentiable variables along the gradient vector.
        optimizer.update(&classifier, along: 𝛁model)
        LazyTensorBarrier()
    }

    Context.local.learningPhase = .inference
    for batch in dataset.validation {
        let (eagerImages, eagerLabels) = (batch.data, batch.label)
        let images = Tensor(copying: eagerImages, to: device)
        let labels = Tensor(copying: eagerLabels, to: device)
        // Compute loss on test set
        let ŷ = classifier(images)
        let loss = softmaxCrossEntropy(logits: ŷ, labels: labels)
        LazyTensorBarrier()
        testStats.update(logits: ŷ, labels: labels, loss: loss)
    }

    print(
        """
        [Epoch \(epoch + 1)] \
        Training Loss: \(String(format: "%.3f", trainStats.averageLoss)), \
        Training Accuracy: \(trainStats.correctGuessCount)/\(trainStats.totalGuessCount) \
        (\(String(format: "%.1f", trainStats.accuracy))%), \
        Test Loss: \(String(format: "%.3f", testStats.averageLoss)), \
        Test Accuracy: \(testStats.correctGuessCount)/\(testStats.totalGuessCount) \
        (\(String(format: "%.3f", testStats.accuracy))%)
        """)
}
