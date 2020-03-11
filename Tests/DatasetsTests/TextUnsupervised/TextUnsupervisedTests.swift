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

import Datasets
import ModelSupport
import TensorFlow
import TextModels
import XCTest

final class TextUnsupervisedTests: XCTestCase {
    func testCreateWikiText103WithBpe() {
        do {
            let gpt2 = try GPT2()
            let dataset = TextUnsupervised(bpe: gpt2.bpe, variant: .wikiText103)

            var totalCount = 0
            for example in dataset.trainingDataset {
                XCTAssertEqual(example.first.shape[0], 299)
                XCTAssertEqual(example.second.shape[0], 299)
                totalCount += 1
            }
            XCTAssertEqual(totalCount, 64)
        } catch {
            XCTFail(error.localizedDescription)
        }
    }

    func testCreateWikiText103WithoutBpe() {
        let dataset = TextUnsupervised(variant: .wikiText103)

        var totalCount = 0
        for example in dataset.trainingDataset {
            XCTAssert(example.first.shape[0] > 336)
            XCTAssert(example.second.shape[0] > 336)
            totalCount += 1
        }
        XCTAssertEqual(totalCount, 128)
    }

    func testCreateWikiText2WithBpe() {
        do {
            let gpt2 = try GPT2()
            let dataset = TextUnsupervised(bpe: gpt2.bpe, variant: .wikiText2)

            var totalCount = 0
            for example in dataset.trainingDataset {
                XCTAssertEqual(example.first.shape[0], 143)
                XCTAssertEqual(example.second.shape[0], 143)
                totalCount += 1
            }
            XCTAssertEqual(totalCount, 64)
        } catch {
            XCTFail(error.localizedDescription)
        }
    }

    func testCreateWikiText2WithoutBpe() {
        let dataset = TextUnsupervised()

        var totalCount = 0
        for example in dataset.trainingDataset {
            XCTAssertEqual(example.first.shape[0], 612)
            XCTAssertEqual(example.second.shape[0], 612)
            totalCount += 1
        }
        XCTAssertEqual(totalCount, 64)
    }
}

extension TextUnsupervisedTests {
    static var allTests = [
        // The WikiText103 dataset is large and should not run in kokoro.
        // Uncomment the following 2 lines to run individually.
        // ("testCreateWikiText103WithBpe", testCreateWikiText103WithBpe),
        // ("testCreateWikiText103WithoutBpe", testCreateWikiText103WithoutBpe),

        ("testCreateWikiText2WithBpe", testCreateWikiText2WithBpe),
        ("testCreateWikiText2WithoutBpe", testCreateWikiText2WithoutBpe),
    ]
}
