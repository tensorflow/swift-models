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

import Batcher
import Foundation
import ModelSupport
import TensorFlow

public enum TextUnsupervisedVariant: String {
  /// - Source: [Einstein AI WikiText-103](
  ///             https://blog.einstein.ai/
  ///             the-wikitext-long-term-dependency-language-modeling-dataset/).
  case wikiText103 = "WikiText103"

  /// Default variant.
  /// - Source: [Einstein AI WikiText-2](
  ///             https://blog.einstein.ai/
  ///             the-wikitext-long-term-dependency-language-modeling-dataset/).
  case wikiText2 = "WikiText2"
}

private protocol TextUnsupervisedVariantDetails {
  var variant: TextUnsupervisedVariant { get set }
  var location: URL { get set }
  var trainingDirectoryName: String { get set }
  var validationDirectoryName: String { get set }
  var filename: String { get set }
  var encodedFileName: String? { get set }
  var fileExtension: String { get set }
}

public struct TextUnsupervised {
  private struct WikiText103Details: TextUnsupervisedVariantDetails {
    var variant = TextUnsupervisedVariant.wikiText103
    var location = URL(string: "https://s3.amazonaws.com/fast-ai-nlp/")!
    var trainingDirectoryName = "train"
    var validationDirectoryName = "test"
    var filename = "wikitext-103"
    var encodedFileName: String? = nil
    var fileExtension = "tgz"
  }

  private struct WikiText2Details: TextUnsupervisedVariantDetails {
    var variant = TextUnsupervisedVariant.wikiText2

    var location = URL(
      string: "https://storage.googleapis.com/s4tf-hosted-binaries/datasets/WikiText2/")!

    var trainingDirectoryName = "train"
    var validationDirectoryName = "test"
    var filename = "wikitext-2"
    var encodedFileName: String? = "wikitext-2-encoded"
    var fileExtension = "tgz"
  }

  public let trainingDataset: LanguageModelDataset<[[Int]]>
  public let validationDataset: LanguageModelDataset<[[Int]]>
  public let bpe: BytePairEncoder?
  public let variant: TextUnsupervisedVariant
  private let variantDetails: TextUnsupervisedVariantDetails

  public init(
    bpe: BytePairEncoder? = nil,
    variant: TextUnsupervisedVariant = TextUnsupervisedVariant.wikiText2,
    trainingBatchSize: Int = 8, validationBatchSize: Int = 4, sequenceLength: Int = 1024,
    trainingDocumentCount: Int = 4, validationDocumentCount: Int = 4
  ) {
    do {
      self.bpe = bpe

      self.variant = variant
      switch variant {
      case .wikiText103:
        let variantDetails = WikiText103Details()
        self.variantDetails = variantDetails
      case .wikiText2:
        let variantDetails = WikiText2Details()
        self.variantDetails = variantDetails
      }

      let localStorageDirectory: URL = FileManager.default.temporaryDirectory
        .appendingPathComponent(
          variant.rawValue, isDirectory: true)
      self.trainingDataset = try TextUnsupervised.loadTraining(
        localStorageDirectory: localStorageDirectory, bpe: bpe,
        variantDetails: variantDetails, batchSize: trainingBatchSize,
        sequenceLength: sequenceLength, documentCount: trainingDocumentCount)
      self.validationDataset = try TextUnsupervised.loadValidation(
        localStorageDirectory: localStorageDirectory, bpe: bpe,
        variantDetails: variantDetails, batchSize: validationBatchSize,
        sequenceLength: sequenceLength, documentCount: validationDocumentCount)
    } catch {
      fatalError("Could not load dataset for \(variant): \(error)")
    }
  }

  private static func downloadIfNotPresent(
    to directory: URL, variantDetails: TextUnsupervisedVariantDetails, downloadEncodedFile: Bool
  ) {
    let downloadPath = directory.appendingPathComponent(variantDetails.variant.rawValue).path
    let directoryExists = FileManager.default.fileExists(atPath: downloadPath)
    let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: downloadPath)
    let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)

    guard !directoryExists || directoryEmpty else { return }

    // Downloads and extracts dataset files.
    let _ = DatasetUtilities.downloadResource(
      filename: (downloadEncodedFile ? variantDetails.encodedFileName! : variantDetails.filename),
      fileExtension: variantDetails.fileExtension,
      remoteRoot: variantDetails.location, localStorageDirectory: directory, extract: true)
  }

  private static func readCSV(in file: URL) throws -> [String] {
    let rawText = try! String(contentsOf: file, encoding: .utf8)
    var rows = rawText.components(separatedBy: "\"\n\"")
    // Removing the initial '"'.
    rows[0] = String(rows[0].dropFirst())
    // Removing the last '"\n'
    rows[rows.indices.last!] = String(rows.last!.dropLast(2))
    return rows
  }

  private static func readEncoded(in file: URL) throws -> [Int] {
    let rawText = try! String(contentsOf: file, encoding: .utf8)
    let rows = rawText.components(separatedBy: "\n")
    var tokens: [Int] = Array()
    for row in rows {
      guard let encoded = Int(row) else { continue }
      tokens.append(encoded)
    }
    return tokens
  }

  private static func embedding(for string: String, bpe: BytePairEncoder) -> [Int] {
    let tokens = bpe.encode(token: string, variant: .gpt2)
    // TODO(michellecasbon): Decide how to prevent OOV or choose a better ID (probably not 0).
    let ids = tokens.map { bpe.vocabulary.id(forToken: $0) ?? 0 }
    return ids
  }

  /// Returns a LanguageModelDataset by processing files specified by 'variantDetails' which
  /// resides in 'directory'.
  ///
  /// Download the files if not present. If bpe is nil which means skip bype pair encoding,
  /// then download the encoded file instead.
  ///
  /// - Parameter name: name of the dataset. Ususally 'train' or 'test'.
  /// - Parameter directory: directory that files are read from.
  /// - Parameter bpe: byte pair encoder used for encoding text.
  /// - Parameter variantDetails: an object containing information of filename, location, etc.
  /// - Parameter batchSize: number of sequences in a batch.
  /// - Parameter sequenceLength: number of characters in a sequence.
  /// - Parameter documentCount: number of documents to proceed. (Refer func readCSV() to see how
  ///   a text file is chunked into documents.)
  private static func loadDirectory(
    named name: String, in directory: URL, bpe: BytePairEncoder?,
    variantDetails: TextUnsupervisedVariantDetails, batchSize: Int, sequenceLength: Int,
    documentCount: Int = 4
  ) throws -> LanguageModelDataset<[[Int]]> {
    precondition(
      bpe != nil || variantDetails.encodedFileName != nil,
      "bpe must be provided when encodedFileName is nil.")
    downloadIfNotPresent(
      to: directory, variantDetails: variantDetails, downloadEncodedFile: bpe == nil)

    var encodedDocs: [[Int]] = []
    if let bpe = bpe {
      let path = directory.appendingPathComponent("\(variantDetails.filename)/\(name).csv")
      let documentsFull = try readCSV(in: path)
      let documents = Array(documentsFull[0..<min(documentCount, documentsFull.count)])
      encodedDocs = documents.concurrentMap { embedding(for: $0, bpe: bpe) }
    } else {
      let pathPrefix = directory.appendingPathComponent(
        "\(variantDetails.encodedFileName!)/\(name)").path
      encodedDocs = (0..<documentCount).map { URL(fileURLWithPath: "\(pathPrefix)/doc_\($0).txt") }
        .concurrentMap
      { try! readEncoded(in: $0) }
    }

    return LanguageModelDataset(
      batchSize: batchSize,
      sequenceLength: sequenceLength,
      numericalizedTexts: encodedDocs,
      lengths: encodedDocs.map { $0.count },
      dropLast: true
    )
  }

  private static func loadTraining(
    localStorageDirectory: URL, bpe: BytePairEncoder?,
    variantDetails: TextUnsupervisedVariantDetails, batchSize: Int, sequenceLength: Int,
    documentCount: Int
  )
    throws
    -> LanguageModelDataset<[[Int]]>
  {
    return try loadDirectory(
      named: variantDetails.trainingDirectoryName, in: localStorageDirectory, bpe: bpe,
      variantDetails: variantDetails, batchSize: batchSize, sequenceLength: sequenceLength,
      documentCount: documentCount)
  }

  private static func loadValidation(
    localStorageDirectory: URL, bpe: BytePairEncoder?,
    variantDetails: TextUnsupervisedVariantDetails, batchSize: Int, sequenceLength: Int,
    documentCount: Int
  )
    throws
    -> LanguageModelDataset<[[Int]]>
  {
    return try loadDirectory(
      named: variantDetails.validationDirectoryName, in: localStorageDirectory, bpe: bpe,
      variantDetails: variantDetails, batchSize: batchSize, sequenceLength: sequenceLength,
      documentCount: documentCount)
  }
}

extension Array {
  fileprivate func concurrentMap<B>(_ transform: @escaping (Element) -> B) -> [B] {
    var res = [B?](repeating: nil, count: count)
    let threadCount = Swift.min(count, 10)
    let q = DispatchQueue(label: "sync queue")
    DispatchQueue.concurrentPerform(iterations: threadCount) { threadId in
      for idx in stride(from: threadId, to: count, by: threadCount) {
        let transformed = transform(self[idx])
        q.sync {
          res[idx] = transformed
        }
      }
    }
    return res.map { $0! }
  }
}
