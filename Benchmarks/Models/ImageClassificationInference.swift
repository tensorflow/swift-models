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
import ImageClassificationModels
import TensorFlow

protocol ImageClassificationModel: Layer where Input == Tensor<Float>, Output == Tensor<Float> {
    init()
    static var preferredInputDimensions: [Int] { get }
    static var outputLabels: Int { get }
}

// TODO: Ease the tight restriction on Batcher data sources to allow for lazy datasets.
class ImageClassificationInference<Model, ClassificationDataset>: Benchmark
where Model: ImageClassificationModel, ClassificationDataset: ImageClassificationDataset,
      ClassificationDataset.SourceDataSet == [TensorPair<Float, Int32>] {
    let testDataset: Batcher<[TensorPair<Float, Int32>]>
    var model: Model
    let batches: Int
    let batchSize: Int

    var exampleCount: Int {
        return batches * batchSize
    }

    init(settings: BenchmarkSettings) {
        self.batches = settings.batches
        self.batchSize = settings.batchSize
        self.model = Model()
        if settings.synthetic {
            let syntheticDataset = SyntheticImageDataset(
                    batchSize: settings.batchSize, batches: settings.batches,
                    labels: Model.outputLabels, dimensions: Model.preferredInputDimensions)
            self.testDataset = syntheticDataset.test
        } else {
            let classificationDataset = ClassificationDataset(batchSize: settings.batchSize)
            self.testDataset = classificationDataset.test
        }
    }

    func run(backend: Backend) -> [Double] {
        let device: Device
        switch backend {
        case .eager: device = Device.defaultTFEager
        case .x10: device = Device.defaultXLA
        }
        model.move(to: device)

        var batchTimings: [Double] = []
        var currentBatch = 0
        for batch in testDataset.sequenced() {
            if (currentBatch >= self.batches) { break }
            let images = batch.first
            let deviceImages: Tensor<Float>
            switch backend {
            case .eager:
                deviceImages = images
            case .x10:
                deviceImages = Tensor(copying: images, to: device)
            }

            let batchTime = time{
                let _ = model(deviceImages)
                LazyTensorBarrier()
            }
            batchTimings.append(batchTime)
            currentBatch += 1
        }
        return batchTimings
    }
}
