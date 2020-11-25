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
import XCTest

@testable import TextModels

class WordSegSemiRingTests: XCTestCase {
  func test_SemiRingAdd() {
    let value: SemiRing =
      SemiRing(logp: 1.0, logr: 2.0) + SemiRing(logp: 3.0, logr: 4.0)
    XCTAssertEqual(value.logp, 3.126928, accuracy: 0.000001)
    XCTAssertEqual(value.logr, 4.126928, accuracy: 0.000001)
  }

  func test_SemiRingInit() {
    let value: SemiRing = SemiRing(logp: 1.0, logr: 2.0)
    XCTAssertEqual(value.logp, 1.0)
    XCTAssertEqual(value.logr, 2.0)
  }

  func test_SemiRingZero() {
    let value: SemiRing = SemiRing.zero
    XCTAssertEqual(value.logp, -Float.infinity)
    XCTAssertEqual(value.logr, -Float.infinity)
  }

  func test_SemiRingAdditiveIdentity() {
    let value: SemiRing = SemiRing.zero + SemiRing(logp: 1.0, logr: 2.0)
    XCTAssertEqual(value.logp, 1.0)
    XCTAssertEqual(value.logr, 2.0)
  }

  func test_SemiRingOne() {
    let value: SemiRing = SemiRing.one
    XCTAssertEqual(value.logp, 0.0)
    XCTAssertEqual(value.logr, -Float.infinity)
  }

  func test_SemiRingMultiplicativeIdentity() {
    let value: SemiRing = SemiRing.one * SemiRing(logp: 1.0, logr: 2.0)
    XCTAssertEqual(value.logp, 1.0)
    XCTAssertEqual(value.logr, 2.0)
  }

  func test_SemiRingMultiply() {
    let value: SemiRing =
      SemiRing(logp: 1.0, logr: 2.0) * SemiRing(logp: 3.0, logr: 4.0)
    XCTAssertEqual(value.logp, 4.0)
    XCTAssertEqual(value.logr, 5.693147, accuracy: 0.000001)
  }

  static var allTests = [
    ("test_SemiRingAdd", test_SemiRingAdd),
    ("test_SemiRingInit", test_SemiRingInit),
    ("test_SemiRingZero", test_SemiRingZero),
    ("test_SemiRingAdditiveIdentity", test_SemiRingAdditiveIdentity),
    ("test_SemiRingOne", test_SemiRingOne),
    ("test_SemiRingMultiplicativeIdentity", test_SemiRingMultiplicativeIdentity),
    ("test_SemiRingMultiply", test_SemiRingMultiply),
  ]
}
