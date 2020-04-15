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

import Batcher
import Datasets
import TensorFlow

// TODO: Ease the tight restriction on Batcher data sources to allow for lazy datasets.
struct ImageClassificationTraining<Model, ClassificationDataset>: Benchmark
where
    Model: ImageClassificationModel, Model.TangentVector.VectorSpaceScalar == Float,
    ClassificationDataset: ImageClassificationDataset,
    ClassificationDataset.SourceDataSet == [TensorPair<Float, Int32>]
{
    let trainingDataset: Batcher<[TensorPair<Float, Int32>]>
    let batches: Int
    let batchSize: Int

    var exampleCount: Int {
        return batches * batchSize
    }

    init(settings: BenchmarkSettings) {
        self.batches = settings.batches
        self.batchSize = settings.batchSize
        if settings.synthetic {
            let syntheticDataset = SyntheticImageDataset(
                    batchSize: settings.batchSize, batches: settings.batches,
                    labels: Model.outputLabels, dimensions: Model.preferredInputDimensions)
            self.trainingDataset = syntheticDataset.training
        } else {
            let classificationDataset = ClassificationDataset(batchSize: settings.batchSize)
            self.trainingDataset = classificationDataset.training
        }
    }

    func run() -> [Double] {
        // Include model and optimizer initialization time in first batch, to be part of warmup.
        var beforeBatch = timestampInMilliseconds()
        var model = Model()
        // TODO: Split out the optimizer as a separate specification.
        let optimizer = SGD(for: model, learningRate: 0.1)
        var batchTimings: [Double] = []
        var currentBatch = 0
        
        Context.local.learningPhase = .training
        while (currentBatch < self.batches) {
            for batch in trainingDataset.sequenced() {
                if (currentBatch >= self.batches) { break }
                let (images, labels) = (batch.first, batch.second)
                
                // Discard remainder batches that are not the same size as the others 
                guard images.shape[0] == self.batchSize else { continue }
                
                let 𝛁model = TensorFlow.gradient(at: model) { model -> Tensor<Float> in
                    let logits = model(images)
                    return softmaxCrossEntropy(logits: logits, labels: labels)
                }
                optimizer.update(&model, along: 𝛁model)
                batchTimings.append(durationInMilliseconds(since: beforeBatch))
                currentBatch += 1
                beforeBatch = timestampInMilliseconds()
            }
        }
        return batchTimings
    }
}
