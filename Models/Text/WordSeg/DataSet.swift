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

import Foundation

public struct DataSet {
  public let training: [CharacterSequence]
  public private(set) var testing: [CharacterSequence]?
  public private(set) var validation: [CharacterSequence]?
  public let alphabet: Alphabet

  private static func load(data: Data) throws -> [String] {
    guard let contents: String = String(data: data, encoding: .utf8) else {
      throw CharacterErrors.nonUtf8Data
    }
    return load(contents: contents)
  }

  private static func load(contents: String) -> [String] {
    var strings = [String]()

    for line in contents.components(separatedBy: .newlines) {
      let stripped: String = line.components(separatedBy: .whitespaces).joined()
      if stripped.isEmpty { continue }
      strings.append(stripped)
    }
    return strings
  }

  private static func makeAlphabet(
    datasets training: [String],
    _ otherSequences: [String]?...,
    eos: String = "</s>",
    eow: String = "</w>",
    pad: String = "</pad>"
  ) -> Alphabet {
    var letters: Set<Character> = []

    for dataset in otherSequences + [training] {
      guard let dataset = dataset else { continue }
      for sentence in dataset {
        for character in sentence {
          letters.insert(character)
        }
      }
    }

    // Sort the letters to make it easier to interpret ints vs letters.
    var sorted = Array(letters)
    sorted.sort()

    return Alphabet(sorted, eos: eos, eow: eow, pad: pad)
  }

  private static func convertDataset(_ dataset: [String], alphabet: Alphabet) throws
    -> [CharacterSequence]
  {
    return try dataset.map { try CharacterSequence(alphabet: alphabet, appendingEoSTo: $0) }
  }
  private static func convertDataset(_ dataset: [String]?, alphabet: Alphabet) throws
    -> [CharacterSequence]?
  {
    if let ds = dataset {
      let tmp: [CharacterSequence] = try convertDataset(ds, alphabet: alphabet)  // Use tmp to disambiguate function
      return tmp
    }
    return nil
  }

  public init?(
    training trainingFile: String,
    validation validationFile: String? = nil,
    testing testingFile: String? = nil
  ) throws {
    let trainingData = try Data(
      contentsOf: URL(fileURLWithPath: trainingFile),
      options: .alwaysMapped)
    let training = try Self.load(data: trainingData)

    var validation: [String]? = nil
    var testing: [String]? = nil

    if let validationFile = validationFile {
      let data = try Data(
        contentsOf: URL(fileURLWithPath: validationFile),
        options: .alwaysMapped)
      validation = try Self.load(data: data)
    }

    if let testingFile = testingFile {
      let data: Data = try Data(
        contentsOf: URL(fileURLWithPath: testingFile),
        options: .alwaysMapped)
      testing = try Self.load(data: data)
    }
    self.alphabet = Self.makeAlphabet(datasets: training, validation, testing)
    self.training = try Self.convertDataset(training, alphabet: self.alphabet)
    self.validation = try Self.convertDataset(validation, alphabet: self.alphabet)
    self.testing = try Self.convertDataset(testing, alphabet: self.alphabet)
  }

  init(training trainingData: Data, validation validationData: Data?, testing testingData: Data?)
    throws
  {
    let training = try Self.load(data: trainingData)
    var validation: [String]? = nil
    var testing: [String]? = nil
    if let validationData = validationData {
      validation = try Self.load(data: validationData)
    }
    if let testingData = testingData {
      testing = try Self.load(data: testingData)
    }

    self.alphabet = Self.makeAlphabet(datasets: training, validation, testing)
    self.training = try Self.convertDataset(training, alphabet: self.alphabet)
    self.validation = try Self.convertDataset(validation, alphabet: self.alphabet)
    self.testing = try Self.convertDataset(testing, alphabet: self.alphabet)
  }
}
