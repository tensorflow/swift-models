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

final class TextInferenceTests: XCTestCase {
    override class func setUp() {
        Context.local.learningPhase = .inference
    }

    func testGPT2_load() {
        do {
            // Load saved model from checkpoint.
            let gpt2 = try GPT2()

            let gpt2Result = try gpt2.generate()
            XCTAssert(gpt2Result.count > 0)
        } catch { XCTFail(error.localizedDescription) }
    }

    func testGPT2_train() {
        do {
            // Do not load from checkpoint.
            let gpt2 = try GPT2(checkpoint: URL(fileURLWithPath: ""))

            let gpt2Result = gpt2.model(Tensor<Int32>(shape: [1, 1], scalars: [0]))
            XCTAssertEqual(gpt2Result.shape, [1, 1, 1])
        } catch { XCTFail(error.localizedDescription) }
    }

    func testGPT2_generate() {
        do {
            // Do not load from checkpoint.
            let gpt2 = try GPT2(checkpoint: URL(fileURLWithPath: ""))

            let gpt2Result = try gpt2.generate()
            XCTAssertEqual(gpt2Result, "<|endoftext|>")
        } catch { XCTFail(error.localizedDescription) }
    }
}

extension TextInferenceTests {
    static var allTests = [
        ("testGPT2_train", testGPT2_train),
        ("testGPT2_generate", testGPT2_generate),
        ("testGPT2_load", testGPT2_load),
    ]
}
