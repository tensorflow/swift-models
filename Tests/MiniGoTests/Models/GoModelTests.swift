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

final class GoModelTests: XCTestCase {
    func testInferenceShape() {
        let modelConfiguration = ModelConfiguration(boardSize: 19)
        let model = GoModel(configuration: modelConfiguration)

        let sampleInput = Tensor<Float>(randomUniform: [2, 19, 19, 17])
        let inference = model.prediction(for: sampleInput)
        let (policy, value) = (inference.policy, inference.value)
        XCTAssertEqual(TensorShape([2, 19 * 19 + 1]), policy.shape)
        XCTAssertEqual(TensorShape([2]), value.shape)
    }
}

extension GoModelTests {
    static var allTests = [
        ("testInferenceShape", testInferenceShape)
    ]
}
