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

/// An Int32-based representation of a string to be used with the WordSeg model.
public struct CharacterSequence: Hashable {
  public let characters: [Int32]
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
    self.characters = [Int32](characters)
    self.eos = alphabet.eos
  }

  public subscript(index: Int32) -> Int32 {
    return characters[Int(index)]
  }

  public subscript(range: Range<Int>) -> ArraySlice<Int32> {
    return characters[range]
  }

  public func tensor(device: Device) -> Tensor<Int32> {
    Tensor<Int32>([self.eos] + characters[0..<characters.count - 1], on: device)
  }

  public var count: Int { return characters.count }
  public var last: Int32? { return characters.last }
}

extension CharacterSequence: CustomStringConvertible {
  public var description: String {
    "\(characters)"
  }
}
