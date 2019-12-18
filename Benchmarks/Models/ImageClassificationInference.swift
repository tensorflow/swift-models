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

protocol ImageClassificationModel: Layer where Input == Tensor<Float>, Output == Tensor<Float> {
    init()
}

extension LeNet: ImageClassificationModel {}

class ImageClassificationInference<Model, ClassificationDataset>: Benchmark
where Model: ImageClassificationModel, ClassificationDataset: ImageClassificationDataset {
    let dataset: ClassificationDataset
    var model: Model
    let images: Tensor<Float>
    let batches: Int
    let batchSize: Int

    var exampleCount: Int {
        return batches * batchSize
    }

    init(settings: BenchmarkSettings, images: Tensor<Float>? = nil) {
        self.batches = settings.batches
        self.batchSize = settings.batchSize
        self.dataset = ClassificationDataset()
        self.model = Model()
        if let providedImages = images {
            self.images = providedImages
        } else {
            self.images = Tensor<Float>(
                randomNormal: [batchSize, 28, 28, 1], mean: Tensor<Float>(0.5),
                standardDeviation: Tensor<Float>(0.1), seed: (0xffeffe, 0xfffe))
        }
    }

    func run() {
        for _ in 0..<batches {
            let _ = model(images)
        }
    }
}
