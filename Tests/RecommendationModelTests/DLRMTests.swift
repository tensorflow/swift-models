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

import TensorFlow
import XCTest

@testable import RecommendationModels

final class DLRMTests: XCTestCase {
    override class func setUp() {
        Context.local.learningPhase = .inference
    }

    func testDLRM() {
        let nDense = 9
        let dimEmbed = 4
        let bottomMLPSize = [8, 4]
        let topMLPSize = [11, 4]
        let batchSize = 10

        let model = DLRM(
            nDense: nDense,
            mSpa: dimEmbed,
            lnEmb: [10, 20],
            lnBot: bottomMLPSize,
            lnTop: topMLPSize)

        let result = model(denseInput: Tensor(ones: [batchSize, nDense]),
                           sparseInput: [Tensor([7, 3, 1, 3, 1, 6, 7, 8, 9, 2]),
                                          Tensor([17, 13, 19, 0, 1, 6, 7, 8, 9, 10])])
        XCTAssertEqual([batchSize], result.shape)
    }

    func testDLRMTraining() {
        let trainingSteps = 400
        let nDense = 9
        let dimEmbed = 4
        let bottomMLPSize = [8, 4]
        let topMLPSize = [11, 4]
        let batchSize = 10

        func lossFunc(predicted: Tensor<Float>, labels: Tensor<Float>) -> Tensor<Float> {
            let difference = predicted - labels
            let squared = difference * difference
            return squared.sum()
        }

        let trainingData = DLRMInput(dense: Tensor(randomNormal: [batchSize, nDense]),
                                      sparse: [Tensor([7, 3, 1, 3, 1, 6, 7, 8, 9, 2]),
                                               Tensor([17, 13, 19, 0, 1, 6, 7, 8, 9, 10])])
        let labels = Tensor<Float>([1,0,0,1,1,1,0,1,0,1])

        // Sometimes DLRM on such a small dataset can get "stuck" in a bad initialization.
        // To ensure a reliable test, we give ourselves a few reinitializations.
        for attempt in 1...5 {
            var model = DLRM(
                nDense: nDense,
                mSpa: dimEmbed,
                lnEmb: [10, 20],
                lnBot: bottomMLPSize,
                lnTop: topMLPSize)
            let optimizer = SGD(for: model, learningRate: 0.1)

            for step in 0...trainingSteps {
                let (loss, grads) = valueWithGradient(at: model) { model in
                    lossFunc(predicted: model(trainingData), labels: labels)
                }
                if step % 50 == 0 {
                    print(step, loss)
                    if round(model(trainingData)) == labels { return }  // Success
                }
                if step > 300 && step % 50 == 0 {
                    print("\n\n-----------------------------------------")
                    print("Step: \(step), loss: \(loss)\nGrads:\n\(grads)\nModel:\n\(model)")
                }
                optimizer.update(&model, along: grads)
            }
            print("Final model outputs (attempt: \(attempt)):\n\(model(trainingData))\nTarget:\n\(labels)")
        }
        XCTFail("Could not perfectly fit a single mini-batch after 5 reinitializations.")
     }
}

extension DLRMTests {
    static var allTests = [
        ("testDLRM", testDLRM),
        ("testDLRMTraining", testDLRMTraining),
    ]
}
