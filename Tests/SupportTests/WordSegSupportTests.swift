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

@testable import TextModels

class WordSegSupportTests: XCTestCase {
  func testAlphabetaConstruct() {
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

  func testCharacterSequenceConstruct() {
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

  func testLexiconConstruct() {
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

  func testLexiconFromSequences() {
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
    ("testAlphabetaConstruct", testAlphabetaConstruct),
    ("testCharacterSequenceConstruct", testCharacterSequenceConstruct),
    ("testLexiconConstruct", testLexiconConstruct),
    ("testLexiconFromSequences", testLexiconFromSequences),
  ]
}
