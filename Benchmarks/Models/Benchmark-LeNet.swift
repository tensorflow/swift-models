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
import ImageClassificationModels

final class LeNetBenchmark {
    let epochs: Int
    let batchSize: Int
    let dataset: MNIST
    var inferenceModel: LeNet!
    var inferenceImages: Tensor<Float>!
    var batches: Int!
    var inferenceBatches: Int!

    init(epochs: Int, batchSize: Int) {
        self.epochs = epochs
        self.batchSize = batchSize
        self.dataset = MNIST(batchSize: batchSize)
    }

    func performTraining() {
        var model = LeNet()
        let optimizer = SGD(for: model, learningRate: 0.1)

        Context.local.learningPhase = .training
        for _ in 1...epochs {
            for i in 0..<dataset.trainingSize / batchSize {
                let x = dataset.trainingImages.minibatch(at: i, batchSize: batchSize)
                let y = dataset.trainingLabels.minibatch(at: i, batchSize: batchSize)
                let 𝛁model = model.gradient { model -> Tensor<Float> in
                    let ŷ = model(x)
                    return softmaxCrossEntropy(logits: ŷ, labels: y)
                }
                optimizer.update(&model, along: 𝛁model)
            }
        }
    }

    func setupInference(_ parameters: BenchmarkVariety) {
        guard case let .inferenceThroughput(batches, batchSize) = parameters else {
            fatalError(
                "Passed the wrong kind of benchmark variety into this setup function: \(parameters)."
            )
        }
        inferenceBatches = batches
        inferenceModel = LeNet()
        inferenceImages = Tensor<Float>(
            randomNormal: [batchSize, 28, 28, 1], mean: Tensor<Float>(0.5),
            standardDeviation: Tensor<Float>(0.1), seed: (0xffeffe, 0xfffe))
    }

    func performInference() {
        for _ in 0..<inferenceBatches {
            let _ = inferenceModel(inferenceImages)
        }
    }
}
