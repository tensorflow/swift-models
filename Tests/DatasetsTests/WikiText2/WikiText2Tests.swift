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

// Original source:
// "WikiText-2"
// Einstein AI
// https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/

import Datasets
import ModelSupport
import TensorFlow
import TextModels
import XCTest

final class WikiText2Tests: XCTestCase {
    func testCreateWikiText2WithBpe() {
        var dataset: WikiText2
        do {
            let gpt2 = try GPT2()
            dataset = WikiText2(bpe: gpt2.bpe)
        } catch {
            // Vocab with one token and empty merge pairs.
            let vocabulary = Vocabulary(tokensToIds: ["<|endoftext|>": 0])
            let mergePairs = [BytePairEncoder.Pair: Int]()
            let bpe = BytePairEncoder(vocabulary: vocabulary, mergePairs: mergePairs)

            dataset = WikiText2(bpe: bpe)
        }

        var totalCount = 0
        for example in dataset.trainingDataset {
            XCTAssert(example.first.shape[0] > 70)
            XCTAssert(example.second.shape[0] > 70)
            totalCount += 1
        }
        XCTAssertEqual(totalCount, 64)
    }

    func testCreateWikiText2WithoutBpe() {
        // Vocab with one token and empty merge pairs.
        let vocabulary = Vocabulary(tokensToIds: ["<|endoftext|>": 0])
        let mergePairs = [BytePairEncoder.Pair: Int]()
        let bpe = BytePairEncoder(vocabulary: vocabulary, mergePairs: mergePairs)

        let dataset = WikiText2(bpe: bpe)

        var totalCount = 0
        for example in dataset.trainingDataset {
            XCTAssert(example.first.shape[0] > 1)
            XCTAssert(example.second.shape[0] > 1)
            totalCount += 1
        }
        XCTAssertEqual(totalCount, 320)
    }
}

extension WikiText2Tests {
    static var allTests = [
        ("testCreateWikiText2WithBpe", testCreateWikiText2WithBpe),
        ("testCreateWikiText2WithoutBpe", testCreateWikiText2WithoutBpe),
    ]
}

