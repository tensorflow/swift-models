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

@testable import VideoClassificationModels

final class VideoClassificationInferenceTests: XCTestCase {
    
    override class func setUp() {
        Context.local.learningPhase = .inference
    }

    func testC3D() {
        let input = Tensor<Float>(
            randomNormal: [1, 12, 256, 256, 3], mean: Tensor<Float>(0.5),
            standardDeviation: Tensor<Float>(0.1), seed: (0xffeffe, 0xfffe))
        let c3d = C3D(classCount: 101)
        let c3dResult = c3d(input)
        XCTAssertEqual(c3dResult.shape, [1, 101])
    }
}

extension VideoClassificationInferenceTests {
    static var allTests = [
        ("testC3D", testC3D)
    ]
}
