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

struct ImageClassificationTraining<Model>
where Model: ImageClassificationModel, Model.TangentVector.VectorSpaceScalar == Float {
    // TODO: (https://github.com/tensorflow/swift-models/issues/206) Datasets should have a common
    // interface to allow for them to be interchangeable in these benchmark cases.
    let dataset: MNIST
    let epochs: Int
    let batchSize: Int

    init(epochs: Int, batchSize: Int) {
        self.epochs = epochs
        self.batchSize = batchSize
        self.dataset = MNIST(batchSize: batchSize)
    }

    func train() {
        var model = Model()
        // TODO: Split out the optimizer as a separate specification.
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
}
