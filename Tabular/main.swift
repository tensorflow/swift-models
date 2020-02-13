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

// Import Dataset
let dataset = BostonHousingDataset()

// Create Model
struct RegressionModel: Layer {
    var layer1 = Dense<Float>(inputSize: 13, outputSize: 64, activation: relu)
    var layer2 = Dense<Float>(inputSize: 64, outputSize: 32, activation: relu)
    var layer3 = Dense<Float>(inputSize: 32, outputSize: 1)
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: layer1, layer2, layer3)
    }
}

var model = RegressionModel()

//Train Model
let optimizer = RMSProp(for: model, learningRate: 0.001)
Context.local.learningPhase = .training

let epochCount = 500
let batchSize = 32
let numberOfBatch = Int(ceil(Double(dataset.numTrainRecords) / Double(batchSize)))
let shuffle = true

func meanAbsoluteError(predictions: Tensor<Float>, truths: Tensor<Float>) -> Float {
    return abs(Tensor<Float>(predictions - truths)).mean().scalarized()
}

print("Starting training...")

for epoch in 1...epochCount {
    var epochLoss: Float = 0
    var epochMAE: Float = 0
    var batchCount: Int = 0
    var batchArray = Array(repeating: false, count: numberOfBatch)
    for batch in 0..<numberOfBatch {
        var r = batch
        if shuffle {
            while true {
                r = Int.random(in: 0..<numberOfBatch)
                if !batchArray[r] {
                    batchArray[r] = true
                    break
                }
            }
        }
        
        let batchStart = r * batchSize
        let batchEnd = min(dataset.numTrainRecords, batchStart + batchSize)
        let (loss, grad) = model.valueWithGradient { (model: RegressionModel) -> Tensor<Float> in
            let logits = model(dataset.xTrain[batchStart..<batchEnd])
            return meanSquaredError(predicted: logits, expected: dataset.yTrain[batchStart..<batchEnd])
        }
        optimizer.update(&model, along: grad)
        
        let logits = model(dataset.xTrain[batchStart..<batchEnd])
        epochMAE += meanAbsoluteError(predictions: logits, truths: dataset.yTrain[batchStart..<batchEnd])
        epochLoss += loss.scalarized()
        batchCount += 1
    }
    epochMAE /= Float(batchCount)
    epochLoss /= Float(batchCount)

    if epoch == epochCount-1 {
        print("MSE: \(epochLoss), MAE: \(epochMAE), Epoch: \(epoch+1)")
    }
}

//Evaluate Model

print("Evaluating model...")

Context.local.learningPhase = .inference

let prediction = model(dataset.xTest)

let evalMse = meanSquaredError(predicted: prediction, expected: dataset.yTest).scalarized()/Float(dataset.numTestRecords)
let evalMae = meanAbsoluteError(predictions: prediction, truths: dataset.yTest)/Float(dataset.numTestRecords)

print("MSE: \(evalMse), MAE: \(evalMae)")
