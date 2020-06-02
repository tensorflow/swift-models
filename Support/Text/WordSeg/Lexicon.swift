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

/// A collection that maps character sequences to logical words.
///
/// In Python implementations, this is sometimes called the string vocabulary
/// (in contrast to the character vocabulary or `Alphabet`, which maps
/// characters to integers).
public struct Lexicon {

  /// A type whose instances represent a sequence of characters.
  public typealias Element = CharacterSequence

  /// A one-to-one mapping between a sequence of characters and unique
  /// integers.
  // TODO(marcrasi): if the value is not used to construct Tensor, switch to Int
  public var dictionary: BijectiveDictionary<CharacterSequence, Int32>

  /// A count of unique logical words in the lexicon.
  public var count: Int { return dictionary.count }

  /// Creates an instance containing a mapping from `sequences` to unique
  /// integers.
  ///
  /// - Parameter sequences: character sequences to compose the lexicon.
  public init<C: Collection>(_ sequences: C) where C.Element == Element {
    self.dictionary = .init(zip(sequences, 0...))
  }

  /// Creates an instance containing a mapping from `sequences` to unique
  /// integers, using `alphabet`. Sequences are truncated at `maxLength` and
  /// only those occurring `minFreq` times are included.
  ///
  /// - Parameter sequences: character sequences to compose the lexicon.
  /// - Parameter alphabet: all characters contained in `sequences`.
  /// - Parameter maxLength: sequence length at which truncation occurs.
  /// - Parameter minFreq: minimum required occurrence of each sequence.
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
