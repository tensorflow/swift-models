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
import XCTest

class WordSegDatasetTests: XCTestCase {
  func testCreateWordSegDataset() {
    do {
      let dataset = try WordSegDataset()
      XCTAssertEqual(dataset.training.count, 7832)
      XCTAssertEqual(dataset.validation!.count, 979)
      XCTAssertEqual(dataset.testing!.count, 979)

      // Check the first example in each set.
      let trainingExample: [Int32] = [
        27, 17, 23, 25, 3, 16, 22, 22, 17, 21, 7, 7, 22,
        10, 7, 4, 17, 17, 13, 29
      ]
      XCTAssertEqual(dataset.training[0].numericalizedText.characters, trainingExample)

      let validationExample: [Int32] = [10, 7, 14, 14, 17, 15, 11, 15, 11, 29]
      XCTAssertEqual(dataset.validation![0].numericalizedText.characters, validationExample)

      let testingExample: [Int32] = [
        14, 7, 22, 15, 7, 21, 7, 7, 11, 8, 11, 5,
        3, 16, 21, 7, 7, 3, 16, 27, 4, 17, 6, 27, 11, 16, 22, 10, 3, 22, 15,
        11, 20, 20, 17, 20, 29
      ]
      XCTAssertEqual(dataset.testing![0].numericalizedText.characters, testingExample)
    } catch {
      XCTFail(error.localizedDescription)
    }
  }

  func testWordSegDatasetLoad() {
    let buffer: [UInt8] = [
      0x61, 0x6c, 0x70, 0x68, 0x61, 0x0a,  // alpha.
    ]

    var dataset: WordSegDataset?
    buffer.withUnsafeBytes { pointer in
      guard let address = pointer.baseAddress else { return }
      let training: Data =
        Data(
          bytesNoCopy: UnsafeMutableRawPointer(mutating: address),
          count: pointer.count, deallocator: .none)
      dataset = try? WordSegDataset(training: training, validation: nil, testing: nil)
    }

    // 'a', 'h', 'l', 'p', '</s>', '</w>', '<pad>'
    XCTAssertEqual(dataset?.alphabet.count, 7)
    XCTAssertEqual(dataset?.training.count, 1)
  }

  static var allTests = [
    ("testCreateWordSegDataset", testCreateWordSegDataset),
    ("testWordSegDatasetLoad", testWordSegDatasetLoad),
  ]
}
