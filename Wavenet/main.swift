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

import Foundation
import TensorFlow

let sampleSize = 100_000
let dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
let model = WavenetModel(
    inputChannels: 1,
    outputChannels: 1,
    batchSize: 1,
    dilations: dilations,
    initialInputWidth: sampleSize,
    filterWidth: 2,
    residualChannels: 32,
    dilationChannels: 32,
    skipChannels: 512
)
let dataset = try WavenetDataset(
    from: "/Users/akshaan.kakar/Downloads",
    variant: WavenetDatasetVariant.vctk,
    audioSampleRate: 16000,
    receptiveField: model.receptiveFieldWidth,
    batchSize: 1,
    sampleSize: sampleSize,
    entropy: SystemRandomNumberGenerator()
)
for (index, epoch) in dataset.training.enumerated() {
    for batch in epoch.inBatches(of: 1) {
        let firstBatch = batch.first!
        let (input, labels) = model.computeInputsAndLabels(firstBatch)
        let preds = model(input)
        let loss = model.loss(predictions: preds, labels: labels)
        if index % 5 == 0 {
            print("Loss = \(loss)")
        }
    }
}
