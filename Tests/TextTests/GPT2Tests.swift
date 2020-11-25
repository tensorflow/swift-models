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

@testable import TextModels

final class GPT2Tests: XCTestCase {
    static var gpt2: GPT2?

    override class func setUp() {
        do {
            // Load saved model from checkpoint.
            gpt2 = try GPT2()
        } catch { 
            XCTFail(error.localizedDescription) 
        }
    }

    func testCheckpointLoadingFailure() {
        XCTAssertThrowsError(try GPT2(checkpoint: URL(fileURLWithPath: "")))
    }

    func testTransformerLMInference() {
        Context.local.learningPhase = .inference

        guard let gpt2 = GPT2Tests.gpt2 else {
            XCTFail("GPT2 failed to initialized")
            return
        }

        let gpt2Result = gpt2.model(Tensor<Int32>(shape: [1, 1], scalars: [0]))
        XCTAssertEqual(gpt2Result.shape, [1, 1, 50257])
    }

    func testGPT2Generate() {
        Context.local.learningPhase = .inference
        
        guard let gpt2 = GPT2Tests.gpt2 else {
            XCTFail("GPT2 failed to initialized")
            return
        }

        do {
            let gpt2Result = try gpt2.generate()
            XCTAssert(gpt2Result.count > 0)
        } catch { 
            XCTFail(error.localizedDescription) 
        }
    }
}

extension GPT2Tests {
    static var allTests = [
        ("testCheckpointLoadingFailure", testCheckpointLoadingFailure),
        ("testTransformerLMInference", testTransformerLMInference),
        ("testGPT2Generate", testGPT2Generate),
    ]
}
