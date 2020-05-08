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

@testable import TextModels

class DataSetTests: XCTestCase {
  func test_DataSetLoad() {
    let buffer: [UInt8] = [
      0x61, 0x6c, 0x70, 0x68, 0x61, 0x0a,  // alpha.
    ]

    var dataset: DataSet?
    buffer.withUnsafeBytes { pointer in
      guard let address = pointer.baseAddress else { return }
      let training: Data =
        Data(
          bytesNoCopy: UnsafeMutableRawPointer(mutating: address),
          count: pointer.count, deallocator: .none)
      dataset = try? DataSet(training: training, validation: nil, testing: nil)
    }

    // 'a', 'h', 'l', 'p', '</s>', '</w>', '<pad>'
    XCTAssertEqual(dataset?.alphabet.count, 7)
    XCTAssertEqual(dataset?.training.count, 1)
  }

  static var allTests = [
    ("test_DataSetLoad", test_DataSetLoad)
  ]
}

class SemiRingTests: XCTestCase {
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

class VocabularyTests: XCTestCase {
  func test_AlphabetaConstruct() {
    let characters: Alphabet = Alphabet(
      [
        "a",
        "b",
        "c",
      ], eos: "</s>", eow: "</w>", pad: "<pad>")
    // a, b, c, EOS, EOW, PAD
    XCTAssertEqual(characters.count, 6)
    XCTAssertEqual(characters.eos, 3)
    XCTAssertEqual(characters.eow, 4)
    XCTAssertEqual(characters.pad, 5)
  }

  func test_CharacterSequenceConstruct() {
    let characters: Alphabet = Alphabet(
      [
        "a",
        "c",
        "t",
      ], eos: "</s>", eow: "</w>", pad: "<pad>")
    let cat: CharacterSequence? = try? CharacterSequence(
      alphabet: characters, appendingEoSTo: "cat")
    XCTAssertNotEqual(cat, nil)
    // FIXME(abdulras) should the EoS be visible?
    XCTAssertEqual(cat?.characters, [Int32(1), Int32(0), Int32(2), characters.eos])

    let bat: CharacterSequence? = try? CharacterSequence(
      alphabet: characters, appendingEoSTo: "bat")
    XCTAssertEqual(bat, nil)
  }

  func test_LexiconConstruct() {
    let characters: Alphabet = Alphabet(
      [
        "a", "b", "e", "g", "h", "l", "m", "p", "t",
      ], eos: "</s>", eow: "</w>", pad: "<pad>")
    let strings: Lexicon = Lexicon([
      try! CharacterSequence(alphabet: characters, appendingEoSTo: "alpha"),
      try! CharacterSequence(alphabet: characters, appendingEoSTo: "beta"),
      try! CharacterSequence(alphabet: characters, appendingEoSTo: "gamma"),
    ])

    XCTAssertEqual(strings.count, 3)
  }

  func test_LexiconFromSequences() {
    let alphabet: Alphabet = Alphabet(
      [
        "a", "b", "e", "g", "h", "l", "m", "p", "t",
      ], eos: "</s>", eow: "</w>", pad: "<pad>")
    let lexicon: Lexicon = Lexicon(
      from: [
        try! CharacterSequence(alphabet: alphabet, appendingEoSTo: "alpha"),
        try! CharacterSequence(alphabet: alphabet, appendingEoSTo: "beta"),
        try! CharacterSequence(alphabet: alphabet, appendingEoSTo: "gamma"),
      ], alphabet: alphabet, maxLength: 5, minFreq: 4)

    XCTAssertEqual(lexicon.count, 3)
  }

  static var allTests = [
    ("test_AlphabetaConstruct", test_AlphabetaConstruct),
    ("test_CharacterSequenceConstruct", test_CharacterSequenceConstruct),
    ("test_LexiconConstruct", test_LexiconConstruct),
    ("test_LexiconFromSequences", test_LexiconFromSequences),
  ]
}
