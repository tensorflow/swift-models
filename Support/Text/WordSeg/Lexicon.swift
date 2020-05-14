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

/// A mapping from characters to logical words.
///
/// In Python implementations, this is sometimes called the String Vocabulary (which is in
/// contrast with the character vocabulary which maps the alphabet to Int32's).
public struct Lexicon {
  public typealias Element = CharacterSequence

  // TODO(marcrasi): if the value is not used to construct Tensor, switch to Int
  public var dictionary: BijectiveDictionary<CharacterSequence, Int32>

  public var count: Int { return dictionary.count }

  public init<C: Collection>(_ sequences: C) where C.Element == Element {
    self.dictionary = .init(zip(sequences, 0...))
  }

  public init(
    from sequences: [CharacterSequence],
    alphabet: Alphabet,
    maxLength: Int,
    minFreq: Int
  ) {
    var histogram: [ArraySlice<Int32>: Int] = [:]

    for sentence in sequences {
      // NOTE: the use of `sentence.count - 1` is to ensure that we ignore the
      // trailing `EoS` marker.
      for i in 0..<sentence.count - 1 {
        for j in 1...maxLength {
          let e = min(i + j, sentence.count - 1)
          // Store strings longer than 2.
          guard e - i > 1 else { continue }
          histogram[sentence[i..<e], default: 0] += 1
        }
      }
    }

    let frequentWordCandidates = histogram.filter { $0.1 >= minFreq }
    let vocab = frequentWordCandidates.map {
      CharacterSequence(alphabet: alphabet, characters: $0.0)
    }

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
      return
        "Unknown character '\(character)' encountered at index \(index) while converting sentence \"\(sentence)\" to a character sequence."
    case .nonUtf8Data:
      return "Non-UTF8 data encountered."
    }
  }
}
