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
import TensorFlow

struct ImageClassificationTraining<Model, ClassificationDataset>: Benchmark
where
    Model: ImageClassificationModel, Model.TangentVector.VectorSpaceScalar == Float,
    ClassificationDataset: ImageClassificationDataset
{
    let dataset: ClassificationDataset
    let epochs: Int

    var exampleCount: Int {
        return epochs * dataset.trainingBatcher.dataset.count
    }

    init(settings: BenchmarkSettings) {
        self.epochs = settings.epochs
        self.dataset = ClassificationDataset(batchSize: settings.batchSize)
    }

    func run() {
        var model = Model()
        // TODO: Split out the optimizer as a separate specification.
        let optimizer = SGD(for: model, learningRate: 0.1)

        Context.local.learningPhase = .training
        for _ in 1...epochs {
            for batch in dataset.trainingBatcher.sequenced() {
                let (images, labels) = (batch.first, batch.second)
                let 𝛁model = TensorFlow.gradient(at: model) { model -> Tensor<Float> in
                    let logits = model(images)
                    return softmaxCrossEntropy(logits: logits, labels: labels)
                }
                optimizer.update(&model, along: 𝛁model)
            }
        }
    }
}