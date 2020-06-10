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

  /// The union of all characters in the included dataset.
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
  private static func load(data: Data) -> [String] {
    let contents = String(decoding: data, as: Unicode.UTF8.self)
    let splitContents = contents.split(separator: "\n", omittingEmptySubsequences: true)
    return splitContents.map { String($0) }
  }

  /// Returns the union of all characters in `phrases`.
  ///
  /// - Parameter eos: the end of sequence marker.
  /// - Parameter eow:the end of word marker.
  /// - Parameter pad: the padding marker.
  private static func makeAlphabet(
    phrases: [String],
    eos: String = "</s>",
    eow: String = "</w>",
    pad: String = "</pad>"
  ) -> Alphabet {
    var letters: Set<Character> = []

    for phrase in phrases {
      for character in phrase {
        if !character.isWhitespace { letters.insert(character) }
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
      let trimmed = data.split(separator: " ", omittingEmptySubsequences: true).joined()
      guard
        let numericalizedText = try? CharacterSequence(
          alphabet: alphabet, appendingEoSTo: trimmed)
      else { continue }
      let phrase = Phrase(
        plainText: data,
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
    let downloadableArchive = DownloadableArchive()
    let localStorageDirectory: URL = DatasetUtilities.defaultDirectory
      .appendingPathComponent("WordSeg", isDirectory: true)

    WordSegDataset.downloadIfNotPresent(
      to: localStorageDirectory, downloadableArchive: downloadableArchive)

    let archiveFileName =
      downloadableArchive
      .location.deletingPathExtension().lastPathComponent
    let archiveDirectory =
      localStorageDirectory
      .appendingPathComponent(archiveFileName)
    let trainingFilePath =
      archiveDirectory
      .appendingPathComponent(downloadableArchive.trainingFilePath).path
    let validationFilePath =
      archiveDirectory
      .appendingPathComponent(downloadableArchive.validationFilePath).path
    let testingFilePath =
      archiveDirectory
      .appendingPathComponent(downloadableArchive.testingFilePath).path

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
    self.trainingPhrases = Self.convertDataset(training, alphabet: self.alphabet)
    self.validationPhrases = Self.convertDataset(validation, alphabet: self.alphabet)
    self.testingPhrases = Self.convertDataset(testing, alphabet: self.alphabet)
  }

  /// Downloads and unpacks `downloadableArchive` to `directory` if it does not
  /// exist locally.
  private static func downloadIfNotPresent(
    to directory: URL, downloadableArchive: DownloadableArchive
  ) {
    let downloadPath = directory.path
    let directoryExists = FileManager.default.fileExists(atPath: downloadPath)
    let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: downloadPath)
    let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)

    guard !directoryExists || directoryEmpty else { return }

    let remoteRoot = downloadableArchive.location.deletingLastPathComponent()
    let filename = downloadableArchive.location.deletingPathExtension().lastPathComponent
    let fileExtension = downloadableArchive.location.pathExtension

    // Downloads and extracts dataset files.
    let _ = DatasetUtilities.downloadResource(
      filename: filename,
      fileExtension: fileExtension,
      remoteRoot: remoteRoot,
      localStorageDirectory: directory, extract: true)
  }
}
