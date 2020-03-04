// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
import TextModels

var gpt = try GPT2()
var model = gpt.model

let dataset = WikiText2()
let trainingBatcher = Batcher(on: dataset.trainingDataset, batchSize: 4)

print("Dataset acquired.")

var optimizer = Adam(for: model, learningRate: 0.02)

print("Starting training...")

for epoch in 1...10 {
    Context.local.learningPhase = .training
    var trainingLossSum: Float = 0
    var trainingBatchCount = 0
    for batch in trainingBatcher.sequenced() {
        let (labels, images) = (batch.first, batch.second)
        // let (loss, gradients) = valueWithGradient(at: model) { model -> Tensor<Float> in
            let logits = model(images)
            let loss = softmaxCrossEntropy(logits: logits, labels: labels)
        // }
        trainingLossSum += loss.scalarized()
        trainingBatchCount += 1
        // optimizer.update(&model, along: gradients)
    }

    print("[Epoch \(epoch)]")
}

/*
for _ in 0..<100 {
    do {
      try print(gpt.generate(), terminator: "")
    } catch {
      continue
    }
}
print()
*/
