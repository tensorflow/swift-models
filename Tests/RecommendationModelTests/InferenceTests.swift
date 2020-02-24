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

final class RecommendationModelInferenceTests: XCTestCase {
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

        let result = model(Tensor(ones: [batchSize, nDense]),
                           sparseInput: [Tensor([7, 3, 1, 3, 1, 6, 7, 8, 9, 2]),
                                          Tensor([17, 13, 19, 0, 1, 6, 7, 8, 9, 10])])
        XCTAssertEqual([batchSize], result.shape)
    }

    func testDLRMTraining() {
        let trainingSteps = 2000
        let nDense = 9
        let dimEmbed = 4
        let bottomMLPSize = [8, 4]
        let topMLPSize = [11, 4]
        let batchSize = 10

        var model = DLRM(
            nDense: nDense,
            mSpa: dimEmbed,
            lnEmb: [10, 20],
            lnBot: bottomMLPSize,
            lnTop: topMLPSize)

        func lossFunc(predicted: Tensor<Float>, labels: Tensor<Float>) -> Tensor<Float> {
            let difference = predicted - labels
            let squared = difference * difference
            return squared.sum()
        }

        let trainingData = DLRMInput(dense: Tensor(ones: [batchSize, nDense]),
                                      sparse: [Tensor([7, 3, 1, 3, 1, 6, 7, 8, 9, 2]),
                                               Tensor([17, 13, 19, 0, 1, 6, 7, 8, 9, 10])])
        let labels = Tensor<Float>([1,0,0,1,1,1,0,1,0,1])

        let optimizer = SGD(for: model, learningRate: 0.002)

        for step in 1...trainingSteps {
            let (loss, grads) = valueWithGradient(at: model) { model in
                lossFunc(predicted: model(trainingData), labels: labels)
            }
            if step % 100 == 0 {
                print(step, loss)
                if loss.scalarized() < 1e-7 {
                    return // Success!
                }
            }
            optimizer.update(&model, along: grads)
        }
        XCTFail("Could not perfectly fit a single mini-batch.")
     }
}

extension RecommendationModelInferenceTests {
    static var allTests = [
        ("testDLRM", testDLRM),
        ("testDLRMTraining", testDLRMTraining)
    ]
}
