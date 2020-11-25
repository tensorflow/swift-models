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

import TensorFlow

/// A sequence of characters represented by integers.
public struct CharacterSequence: Hashable {

  /// Represents an ordered sequence of characters.
  public let characters: [Int32]

  /// A marker denoting the end of the sequence.
  private let eos: Int32

  /// Creates an empty instance without meaningful contents.
  public init(_debug: Int) {
    self.characters = []
    self.eos = -1
  }

  /// Creates a sequence from `string`, using `alphabet`, appended with the
  /// end marker.
  ///
  /// - Throws: `CharacterErrors.unknownCharacter` if `string` contains a
  ///   character that does not exist in `alphabet`.
  public init(alphabet: Alphabet, appendingEoSTo string: String) throws {
    var characters = [Int32]()
    characters.reserveCapacity(string.count + 1)
    for (index, character) in string.enumerated() {
      guard let value = alphabet[String(character)] else {
        throw CharacterErrors.unknownCharacter(character: character, index: index, sentence: string)
      }
      characters.append(value)
    }
    characters.append(alphabet.eos)
    self.init(alphabet: alphabet, characters: characters)
  }

  /// Creates a sequence from `characters` and sets the end marker from
  /// `alphabet`.
  ///
  /// - Note: Assumes `characters` contains an end marker.
  private init(alphabet: Alphabet, characters: [Int32]) {
    self.characters = characters
    self.eos = alphabet.eos
  }

  /// Creates a sequence from `characters` and sets the end marker from
  /// `alphabet`.
  ///
  /// - Note: Assumes `characters` contains an end marker.
  public init(alphabet: Alphabet, characters: ArraySlice<Int32>) {
    self.characters = [Int32](characters)
    self.eos = alphabet.eos
  }

  /// Accesses the `index`th character.
  public subscript(index: Int32) -> Int32 {
    return characters[Int(index)]
  }

  /// Accesses characters within `range`.
  public subscript(range: Range<Int>) -> ArraySlice<Int32> {
    return characters[range]
  }

  /// Count of characters in the sequence, including the end marker.
  public var count: Int { return characters.count }

  /// The last character in the sequence, if `characters` is not empty.
  ///
  /// - Note: This is usually the end marker.
  public var last: Int32? { return characters.last }

  /// Representation for character generation on `device`, with the end marker
  /// moved to the beginning.
  public func tensor(device: Device) -> Tensor<Int32> {
    Tensor<Int32>([self.eos] + characters[0..<characters.count - 1], on: device)
  }
}

extension CharacterSequence: CustomStringConvertible {

  /// A string representation of the integers in the character sequence.
  public var description: String {
    "\(characters)"
  }
}

/// An error that can be encountered when processing characters.
public enum CharacterErrors: Error {
  case unknownCharacter(character: Character, index: Int, sentence: String)
}

extension CharacterErrors: CustomStringConvertible {

  /// A description of the error with all included details.
  public var description: String {
    switch self {
    case let .unknownCharacter(character, index, sentence):
      return
        "Unknown character '\(character)' encountered at index \(index) while converting sentence \"\(sentence)\" to a character sequence."
    }
  }
}
