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
import XCTest

@testable import MiniGo

final class MCTSModelBasedPredictorTests: XCTestCase {

    struct MockModel: InferenceModel {
        func prediction(for: Tensor<Float>) -> GoModelOutput {
            return GoModelOutput(
                policy: Tensor<Float>(rangeFrom: 0, to: 19 * 19 + 1, stride:1),
                value: Tensor<Float>(shape: [1], scalars: [0.9]),
                logits: Tensor<Float>(randomUniform: [1, 19 * 19 + 1]))  // Not used.
        }
    }

    func testPrediction() {
        let configuration = GameConfiguration(size: 19, komi: 0.1)
        let boardState = BoardState(gameConfiguration: configuration)

        let predictor = MCTSModelBasedPredictor(boardSize: 19, model: MockModel())
        let prediction = predictor.prediction(for: boardState)

        XCTAssertEqual(0.9, prediction.rewardForNextPlayer)
        XCTAssertEqual(Float(19 * 19), prediction.distribution.pass)
        for x in 0..<19 {
            for y in 0..<19 {
                XCTAssertEqual(ShapedArraySlice(Float(x * 19 + y)), prediction.distribution.positions[x][y])
            }
        }
    }
}

extension MCTSModelBasedPredictorTests {
    static var allTests = [
        ("testPrediction", testPrediction),
    ]
}
