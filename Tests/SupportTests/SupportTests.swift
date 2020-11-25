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

import XCTest
import ModelSupport

final class SupportTests: XCTestCase {
    func testBijectiveDictionaryConstruct() {
      let _: BijectiveDictionary<Int, String> =
          BijectiveDictionary([1: "one", 2: "two"])

      let dictionary: [Int:String] = [1: "one", 2: "two"]
      let _: BijectiveDictionary<Int, String> =
          BijectiveDictionary(dictionary)

      let array: [(Int, String)] = [(1, "one"), (2, "two")]
      let _: BijectiveDictionary<Int, String> =
          BijectiveDictionary(array)
    }

    func testBijectiveDictionaryCount() {
      let map: BijectiveDictionary<Int, String> =
          BijectiveDictionary([1: "one", 2: "two"])
      XCTAssertEqual(map.count, 2)
    }

    func testBijectiveDictionarySubscript() {
      let map: BijectiveDictionary<Int, String> =
          BijectiveDictionary([1: "one", 2: "two"])
      XCTAssertEqual(map[1], "one")
      XCTAssertEqual(map.key("one"), 1)
    }

    func testBijectiveDictionaryDeletion() {
      var map: BijectiveDictionary<Int, String> =
          BijectiveDictionary([1: "one", 2: "two"])
      XCTAssertEqual(map.count, 2)

      map[2] = nil

      XCTAssertEqual(map.count, 1)
      XCTAssertEqual(map[1], "one")
      XCTAssertEqual(map.key("one"), 1)
    }

    func testBijectiveDictionaryRemapping() {
      // 1 -> "two", 2 -> "four"
      var map: BijectiveDictionary<Int, String> =
          BijectiveDictionary([1: "two", 2: "four"])
      XCTAssertEqual(map.count, 2)

      // 1 -> "three", 2 -> "four"
      map[1] = "three"

      XCTAssertEqual(map.count, 2)
      XCTAssertEqual(map[1], "three")
      XCTAssertEqual(map[2], "four")

      // 2 -> "three"
      map[2] = "three"

      XCTAssertEqual(map.count, 1)
      XCTAssertEqual(map[2], "three")
    }

    static var allTests = [
        ("testBijectiveDictionaryConstruct", testBijectiveDictionaryConstruct),
        ("testBijectiveDictionaryCount", testBijectiveDictionaryCount),
        ("testBijectiveDictionarySubscript", testBijectiveDictionarySubscript),
        ("testBijectiveDictionaryDeletion", testBijectiveDictionaryDeletion),
        ("testBijectiveDictionaryRemapping", testBijectiveDictionaryRemapping),
    ]
}

