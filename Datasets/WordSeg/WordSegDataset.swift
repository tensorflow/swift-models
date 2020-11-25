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

  /// The training data.
  public let trainingPhrases: [Phrase]

  /// The test data.
  public private(set) var testingPhrases: [Phrase]

  /// The validation data.
  public private(set) var validationPhrases: [Phrase]

  /// A mapping between characters used in the dataset and densely-packed integers
  public let alphabet: Alphabet

  /// A pointer to source data.
  private struct DownloadableArchive {

    /// A [web resource](https://en.wikipedia.org/wiki/Web_resource) that can be unpacked
    /// into data files described by other properties of `self`. 
    let location = URL(string: "https://s3.eu-west-2.amazonaws.com/k-kawakami/seg.zip")!

    /// The path to the test data within the unpacked archive.
    let testingFilePath = "br/br-text/te.txt"

    /// The path to the training data within the unpacked archive.
    let trainingFilePath = "br/br-text/tr.txt"

    /// The path to the validation data within the unpacked archive.
    let validationFilePath = "br/br-text/va.txt"
  }

  /// Returns phrases parsed from `data` in UTF8, separated by newlines.
  private static func load(data: Data) -> [Substring] {
    let contents = String(decoding: data, as: Unicode.UTF8.self)
    let splitContents = contents.split(separator: "\n", omittingEmptySubsequences: true)
    return splitContents
  }

  /// Returns the union of all characters in `phrases`.
  ///
  /// - Parameter eos: the end of sequence marker.
  /// - Parameter eow:the end of word marker.
  /// - Parameter pad: the padding marker.
  private static func makeAlphabet(
    phrases: [Substring],
    eos: String = "</s>",
    eow: String = "</w>",
    pad: String = "</pad>"
  ) -> Alphabet {
    let letters = Set(phrases.joined().lazy.filter { !$0.isWhitespace })

    // Sort the letters to make it easier to interpret ints vs letters.
    let sorted = Array(letters).sorted()

    return Alphabet(sorted, eos: eos, eow: eow, pad: pad)
  }

  /// Numericalizes `dataset` with the mapping in `alphabet`, to be used with the
  /// WordSeg model.
  ///
  /// - Note: Omits any phrase that cannot be converted to `CharacterSequence`.
  private static func numericalizeDataset(_ dataset: [Substring], alphabet: Alphabet)
    -> [Phrase]
  {
    var phrases = [Phrase]()

    for data in dataset {
      let trimmed = data.split(separator: " ", omittingEmptySubsequences: true).joined()
      guard
        let numericalizedText = try? CharacterSequence(
          alphabet: alphabet, appendingEoSTo: trimmed)
      else { continue }
      let phrase = Phrase(
        plainText: String(data),
        numericalizedText: numericalizedText)
      phrases.append(phrase)
    }

    return phrases
  }

  /// Creates an instance containing phrases from the reference archive.
  ///
  /// - Throws: an error in the Cocoa domain, if the default training file
  ///   cannot be read.
  public init() throws {
    let source = DownloadableArchive()
    let localStorageDirectory: URL = DatasetUtilities.defaultDirectory
      .appendingPathComponent("WordSeg", isDirectory: true)

    Self.downloadIfNotPresent(
      to: localStorageDirectory, source: source)

    let archiveFileName = source.location.deletingPathExtension().lastPathComponent
    let archiveDirectory =
      localStorageDirectory
      .appendingPathComponent(archiveFileName)
    let trainingFilePath =
      archiveDirectory
      .appendingPathComponent(source.trainingFilePath).path
    let validationFilePath =
      archiveDirectory
      .appendingPathComponent(source.validationFilePath).path
    let testingFilePath =
      archiveDirectory
      .appendingPathComponent(source.testingFilePath).path

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

    let validationData = try Data(
      contentsOf: URL(fileURLWithPath: validationFile ?? "/dev/null"),
      options: .alwaysMapped)

    let testingData = try Data(
      contentsOf: URL(fileURLWithPath: testingFile ?? "/dev/null"),
      options: .alwaysMapped)

    self.init(
      training: trainingData, validation: validationData, testing: testingData)
  }

  /// Creates an instance containing phrases from `trainingData`, and
  /// optionally `validationData` and `testingData`.
  public init(
    training trainingData: Data, validation validationData: Data?, testing testingData: Data?
  ) {
    let training = Self.load(data: trainingData)
    let validation = Self.load(data: validationData ?? Data())
    let testing = Self.load(data: testingData ?? Data())

    self.alphabet = Self.makeAlphabet(phrases: training + validation + testing)
    self.trainingPhrases = Self.numericalizeDataset(training, alphabet: self.alphabet)
    self.validationPhrases = Self.numericalizeDataset(validation, alphabet: self.alphabet)
    self.testingPhrases = Self.numericalizeDataset(testing, alphabet: self.alphabet)
  }

  /// Downloads and unpacks `source` to `directory` if it does not
  /// exist locally.
  private static func downloadIfNotPresent(
    to directory: URL, source: DownloadableArchive
  ) {
    let downloadPath = directory.path
    let directoryExists = FileManager.default.fileExists(atPath: downloadPath)
    let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: downloadPath)
    let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)

    guard !directoryExists || directoryEmpty else { return }

    let remoteRoot = source.location.deletingLastPathComponent()
    let filename = source.location.deletingPathExtension().lastPathComponent
    let fileExtension = source.location.pathExtension

    // Downloads and extracts dataset files.
    let _ = DatasetUtilities.downloadResource(
      filename: filename,
      fileExtension: fileExtension,
      remoteRoot: remoteRoot,
      localStorageDirectory: directory, extract: true)
  }
}
