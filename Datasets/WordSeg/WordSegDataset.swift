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
import ModelSupport

/// A dataset targeted at the problem of word segmentation.
///
/// The reference archive was published in the paper "Learning to Discover,
/// Ground, and Use Words with Segmental Neural Language Models" by Kazuya
/// Kawakami, Chris Dyer, and Phil Blunsom:
/// https://www.aclweb.org/anthology/P19-1645.pdf.
public struct WordSegDataset {

  /// The text used for training.
  public let trainingPhrases: [Phrase]

  /// The text used for testing.
  public private(set) var testingPhrases: [Phrase]?

  /// The text used for validation.
  public private(set) var validationPhrases: [Phrase]?

  /// The union of all characters in the included dataset.
  public let alphabet: Alphabet

  /// A pointer to source data.
  private struct ReferenceArchive {

    /// The location of the archive.
    var location = URL(string: "https://s3.eu-west-2.amazonaws.com/k-kawakami/seg.zip")!

    /// The path to the test source.
    var testingFilePath = "br/br-text/te.txt"

    /// The path to the training source.
    var trainingFilePath = "br/br-text/tr.txt"

    /// The path to the validation source.
    var validationFilePath = "br/br-text/va.txt"
  }

  /// Returns the text of all phrases parsed from `data` in UTF8.
  private static func load(data: Data) -> [String] {
    guard let contents: String = String(data: data, encoding: .utf8) else {
      return []
    }
    return load(contents: contents)
  }

  /// Returns the text of all phrases from `contents`.
  private static func load(contents: String) -> [String] {
    var strings = [String]()

    for line in contents.components(separatedBy: .newlines) {
      let trimmed = line.trimmingCharacters(in: .whitespaces)
      if trimmed.isEmpty { continue }
      strings.append(trimmed)
    }
    return strings
  }

  /// Returns the union of all characters in `training` and `otherSequences`.
  ///
  /// - Parameter eos: text to be used as the end of sequence marker.
  /// - Parameter eow: text to be used as the end of word marker.
  /// - Parameter pad: text to be used as the padding marker.
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
          if !character.isWhitespace { letters.insert(character) }
        }
      }
    }

    // Sort the letters to make it easier to interpret ints vs letters.
    var sorted = Array(letters)
    sorted.sort()

    return Alphabet(sorted, eos: eos, eow: eow, pad: pad)
  }

  /// Returns phrases from `dataset`, using `alphabet`, to be used with the
  /// WordSeg model.
  ///
  /// - Note: Omits any part of the dataset that cannot be converted to
  ///   `CharacterSequence`.
  private static func convertDataset(_ dataset: [String], alphabet: Alphabet)
    -> [Phrase]
  {
    var phrases = [Phrase]()

    for data in dataset {
      let trimmed = data.components(separatedBy: .whitespaces).joined()
      guard let numericalizedText = try? CharacterSequence(
          alphabet: alphabet, appendingEoSTo: trimmed) else { continue }
      let phrase = Phrase(
        plainText: data,
        numericalizedText: numericalizedText)
      phrases.append(phrase)
    }

    return phrases
  }

  /// Creates an instance containing phrases from the default location.
  ///
  /// - Throws: an error in the Cocoa domain, if the default training file
  ///   cannot be read.
  public init() throws {
    let referenceArchive = ReferenceArchive()
    let localStorageDirectory: URL = DatasetUtilities.defaultDirectory
      .appendingPathComponent("WordSeg", isDirectory: true)

    WordSegDataset.downloadIfNotPresent(to: localStorageDirectory, referenceArchive: referenceArchive)

    let archiveFileName =
      referenceArchive
      .location.deletingPathExtension().lastPathComponent
    let archiveDirectory =
      localStorageDirectory
      .appendingPathComponent(archiveFileName)
    let trainingFilePath =
      archiveDirectory
      .appendingPathComponent(referenceArchive.trainingFilePath).path
    let validationFilePath =
      archiveDirectory
      .appendingPathComponent(referenceArchive.validationFilePath).path
    let testingFilePath =
      archiveDirectory
      .appendingPathComponent(referenceArchive.testingFilePath).path

    try self.init(
      training: trainingFilePath, validation: validationFilePath,
      testing: testingFilePath)
  }

  /// Creates an instance containing phrases from `trainingFile`, and
  /// optionally `validationFile` and `testingFile`.
  ///
  /// - Throws: an error in the Cocoa domain, if `trainingFile` cannot be
  ///   read.
  public init(
    training trainingFile: String,
    validation validationFile: String? = nil,
    testing testingFile: String? = nil
  ) throws {
    let trainingData = try Data(
      contentsOf: URL(fileURLWithPath: trainingFile),
      options: .alwaysMapped)
    let training = Self.load(data: trainingData)

    let validation: [String]
    let testing: [String]

    if let validationFile = validationFile {
      let data = try Data(
        contentsOf: URL(fileURLWithPath: validationFile),
        options: .alwaysMapped)
      validation = Self.load(data: data)
    } else {
      validation = [String]()
    }

    if let testingFile = testingFile {
      let data: Data = try Data(
        contentsOf: URL(fileURLWithPath: testingFile),
        options: .alwaysMapped)
      testing = Self.load(data: data)
    } else {
      testing = [String]()
    }

    self.alphabet = Self.makeAlphabet(datasets: training, validation, testing)
    self.trainingPhrases = Self.convertDataset(training, alphabet: self.alphabet)
    self.validationPhrases = Self.convertDataset(validation, alphabet: self.alphabet)
    self.testingPhrases = Self.convertDataset(testing, alphabet: self.alphabet)
  }

  /// Creates an instance containing phrases from `trainingData`, and
  /// optionally `validationData` and `testingData`.
  public init(
    training trainingData: Data, validation validationData: Data?, testing testingData: Data?
  )
  {
    let training = Self.load(data: trainingData)
    let validation: [String]
    let testing: [String]
    if let validationData = validationData {
      validation = Self.load(data: validationData)
    } else {
      validation = [String]()
    }
    if let testingData = testingData {
      testing = Self.load(data: testingData)
    } else {
      testing = [String]()
    }

    self.alphabet = Self.makeAlphabet(datasets: training, validation, testing)
    self.trainingPhrases = Self.convertDataset(training, alphabet: self.alphabet)
    self.validationPhrases = Self.convertDataset(validation, alphabet: self.alphabet)
    self.testingPhrases = Self.convertDataset(testing, alphabet: self.alphabet)
  }

  /// Downloads and unpacks `referenceArchive` to `directory` if it does not
  /// exist locally.
  private static func downloadIfNotPresent(
    to directory: URL, referenceArchive: ReferenceArchive
  ) {
    let downloadPath = directory.path
    let directoryExists = FileManager.default.fileExists(atPath: downloadPath)
    let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: downloadPath)
    let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)

    guard !directoryExists || directoryEmpty else { return }

    let remoteRoot = referenceArchive.location.deletingLastPathComponent()
    let filename = referenceArchive.location.deletingPathExtension().lastPathComponent
    let fileExtension = referenceArchive.location.pathExtension

    // Downloads and extracts dataset files.
    let _ = DatasetUtilities.downloadResource(
      filename: filename,
      fileExtension: fileExtension,
      remoteRoot: remoteRoot,
      localStorageDirectory: directory, extract: true)
  }
}
