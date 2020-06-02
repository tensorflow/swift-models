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

/// A collection that maps individual characters to an integer representation.
///
/// In Python implementations, this is sometimes called the character vocabulary.
///
/// - Note: We map from String in order to support multi-character metadata sequences such as `</s>`.
public struct Alphabet {

  /// A type whose instances represent a character.
  public typealias Element = String

  /// A one-to-one mapping between a set of characters and a unique integer.
  public var dictionary: BijectiveDictionary<String, Int32>

  /// A marker denoting the end of a sequence.
  public let eos: Int32

  /// A marker denoting the end of a word.
  public let eow: Int32

  /// A marker used for padding inside a sequence.
  public let pad: Int32

  /// Creates an instance containing a mapping from `letters` to unique
  /// integers, including markers.
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

  /// Creates an instance containing a mapping from `letters` to unique
  /// integers, including markers.
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

  /// A count of the characters in the alphabet, including markers.
  public var count: Int { return dictionary.count }

  /// Accesses the `key`th element.
  public subscript(key: String) -> Int32? {
    return dictionary[key]
  }
}
