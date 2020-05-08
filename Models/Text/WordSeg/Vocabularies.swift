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
import ModelSupport

/// Alphabet maps from characters in a string to Int32 representations.
///
/// Note: we map from String in order to support multi-character metadata sequences such as </s>.
///
/// In Python implementations, this is sometimes called the character vocabulary.
public struct Alphabet {
  public typealias Element = String

  var dictionary: BijectiveDictionary<String, Int32>

  let eos: Int32
  let eow: Int32
  let pad: Int32

  public init<C: Collection>(_ letters: C, eos: String, eow: String, pad: String)
      where C.Element == Character {
    self.dictionary = .init(zip(letters.lazy.map { String($0) }, 0...))

    self.eos = Int32(self.dictionary.count)
    self.dictionary[eos] = self.eos

    self.eow = Int32(self.dictionary.count)
    self.dictionary[eow] = self.eow

    self.pad = Int32(self.dictionary.count)
    self.dictionary[pad] = self.pad
  }

  public init<C: Collection>(_ letters: C, eos: String, eow: String, pad: String)
      where C.Element == Element {
    self.dictionary = .init(zip(letters.lazy.map { String($0) }, 0...))

    self.eos = Int32(self.dictionary.count)
    self.dictionary[eos] = self.eos

    self.eow = Int32(self.dictionary.count)
    self.dictionary[eow] = self.eow

    self.pad = Int32(self.dictionary.count)
    self.dictionary[pad] = self.pad
  }

  var count: Int { return dictionary.count }

  subscript(key: String) -> Int32? {
    return dictionary[key]
  }
}

/// An Int32-based representation of a string to be used with the WordSeg model.
public struct CharacterSequence: Hashable {
  let characters: [Int32]
  private let eos: Int32

  public init(_debug: Int) {
    self.characters = []
    self.eos = -1
  }

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

  private init(alphabet: Alphabet, characters: [Int32]) {
    self.characters = characters
    self.eos = alphabet.eos
  }

  public init(alphabet: Alphabet, characters: ArraySlice<Int32>) {
    self.characters = Array<Int32>(characters)
    self.eos = alphabet.eos
  }

  subscript(index: Int32) -> Int32 {
    return characters[Int(index)]
  }

  subscript(range: Range<Int>) -> ArraySlice<Int32> {
    return characters[range]
  }

  public var count: Int { return characters.count }
  var last: Int32? { return characters.last }
  var tensor: Tensor<Int32> {
    Tensor<Int32>([self.eos] + characters[0 ..< characters.count - 1])
  }
}

extension CharacterSequence: CustomStringConvertible {
  public var description: String {
    "\(characters)"
  }
}

/// A mapping from characters to logical words.
///
/// In Python implementations, this is sometimes called the String Vocabulary (which is in
/// contrast with the character vocabulary which maps the alphabet to Int32's).
public struct Lexicon {
  public typealias Element = CharacterSequence

  // TODO(marcrasi): if the value is not used to construct Tensor, switch to Int
  var dictionary: BijectiveDictionary<CharacterSequence, Int32>

  var count: Int { return dictionary.count }

  public init<C: Collection>(_ sequences: C) where C.Element == Element {
    self.dictionary = .init(zip(sequences, 0...))
  }

  public init(
    from sequences: [CharacterSequence],
    alphabet: Alphabet,
    maxLength: Int,
    minFreq: Int
  ) {
    var histogram: [ArraySlice<Int32>:Int] = [:]

    for sentence in sequences {
      // NOTE: the use of `sentence.count - 1` is to ensure that we ignore the
      // trailing `EoS` marker.
      for i in 0 ..< sentence.count - 1 {
        for j in 1 ... maxLength {
          let e = min(i + j, sentence.count - 1)
          // Store strings longer than 2.
          guard e - i > 1 else { continue }
          histogram[sentence[i ..< e], default: 0] += 1
        }
      }
    }

    let frequentWordCandidates = histogram.filter { $0.1 >= minFreq }
    let vocab = frequentWordCandidates.map { CharacterSequence(alphabet: alphabet, characters: $0.0) }

    self.init(vocab)
  }
}

public enum CharacterErrors: Error {
  case unknownCharacter(character: Character, index: Int, sentence: String)
  case nonUtf8Data
}

extension CharacterErrors: CustomStringConvertible {
  public var description: String {
    switch self {
    case let .unknownCharacter(character, index, sentence):
      return "Unknown character '\(character)' encountered at index \(index) while converting sentence \"\(sentence)\" to a character sequence."
    case .nonUtf8Data:
      return "Non-UTF8 data encountered."
    }
  }
}
