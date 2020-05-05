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
//
// Adapted from: https://gist.github.com/eaplatanios/5163c8d503f9e56f11b5b058fb041d62
// Changes:
// - Rename `Architecture` to `BERTClassifier`.
// - In `CoLA.update`:
//   - Change `Architecture.classify` to `BERTClassifier.callAsFunction`.
//   - Change `softmaxCrossEntropy` to `sigmoidCrossEntropy`.

import Foundation
import ModelSupport
import TensorFlow

/// A `TextBatch` with the corresponding labels.
public typealias LabeledTextBatch = (data: TextBatch, label: Tensor<Int32>)

/// CoLA example.
public struct CoLAExample {
  /// The unique identifier representing the `Example`.
  public let id: String
  /// The text of the `Example`.
  public let sentence: String
  /// The label of the `Example`.
  public let isAcceptable: Bool?

  /// Creates an instance from `id`, `sentence` and `isAcceptable`.
  public init(id: String, sentence: String, isAcceptable: Bool?) {
    self.id = id
    self.sentence = sentence
    self.isAcceptable = isAcceptable
  }
}

public struct CoLA<Entropy: RandomNumberGenerator> {
  /// The directory where the dataset will be downloaded
  public let directoryURL: URL
  /// The type of the labeled samples.
  public typealias Samples = LazyMapSequence<[CoLAExample], LabeledTextBatch>
  /// The training texts.
  public let trainingExamples: Samples
  /// The validation texts.
  public let validationExamples: Samples
  /// The test texts. 
//  public let testExamples: [Example]
    
  /// The sequence length to which every sentence will be padded.
  public let maxSequenceLength: Int
  /// The batch size.
  public let batchSize: Int
  /// The type of the collection of batches.
  public typealias Batches = Slices<Sampling<Samples, ArraySlice<Int>>>
  /// The type of the training seauence of epochs.
  public typealias TrainEpochs = LazyMapSequence<TrainingEpochs<Samples, Entropy>, 
    LazyMapSequence<Batches, LabeledTextBatch>>
  
  //public typealias DevDataIterator = GroupedIterator<MapIterator<ExampleIterator, DataBatch>>
  //public typealias TestDataIterator = DevDataIterator

  public var trainingEpochs: TrainEpochs
  public var validationBatches: LazyMapSequence<Slices<Samples>, LabeledTextBatch>
//  public var testDataIterator: TestDataIterator
    
  /// The url from which to download the dataset.
  private let url: URL = URL(
    string: String(
      "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/"
      + "o/data%2FCoLA.zip?alt=media&token=46d5e637-3411-4188-bc44-5809b5bfb5f4"))!
}

// Data
extension CoLA {
  internal static func load(fromFile fileURL: URL, isTest: Bool = false) throws -> [CoLAExample] {
    let lines = try parse(tsvFileAt: fileURL)

    if isTest {
      // The test data file has a header.
      return lines.dropFirst().enumerated().map { (i, lineParts) in
        CoLAExample(id: lineParts[0], sentence: lineParts[1], isAcceptable: nil)
      }
    }

    return lines.enumerated().map { (i, lineParts) in
      CoLAExample(id: lineParts[0], sentence: lineParts[3], isAcceptable: lineParts[1] == "1")
    }
  }
}

internal func parse(tsvFileAt fileURL: URL) throws -> [[String]] {
    try Data(contentsOf: fileURL).withUnsafeBytes {
        $0.split(separator: UInt8(ascii: "\n")).map {
            $0.split(separator: UInt8(ascii: "\t"), omittingEmptySubsequences: false)
                .map { String(decoding: UnsafeRawBufferPointer(rebasing: $0), as: UTF8.self) }
        }
    }
}

extension CoLA {
  /// Creates an instance in `taskDirectoryURL` with batches of size `batchSize`
  /// by `maximumSequenceLength`.
  ///
  /// - Parameters:
  ///   - entropy: a source of randomness used to shuffle sample ordering. It
  ///     will be stored in `self`, so if it is only pseudorandom and has value
  ///     semantics, the sequence of epochs is determinstic and not dependent on
  ///     other operations.
  ///   - exampleMap: a transform that processes `Example` in `LabeledTextBatch`.
  public init(
    taskDirectoryURL: URL,
    maxSequenceLength: Int,
    batchSize: Int,
    entropy: Entropy,
    exampleMap: @escaping (CoLAExample) -> LabeledTextBatch
  ) throws {
    self.directoryURL = taskDirectoryURL.appendingPathComponent("CoLA")
    let dataURL = directoryURL.appendingPathComponent("data")
    let compressedDataURL = dataURL.appendingPathComponent("downloaded-data.zip")

    // Download the data, if necessary.
    try download(from: url, to: compressedDataURL)

    // Extract the data, if necessary.
    let extractedDirectoryURL = compressedDataURL.deletingPathExtension()
    if !FileManager.default.fileExists(atPath: extractedDirectoryURL.path) {
      try extract(zipFileAt: compressedDataURL, to: extractedDirectoryURL)
    }

    #if false
      // FIXME: Need to generalize `DatasetUtilities.downloadResource` to accept
      // arbitrary full URLs instead of constructing full URL from filename and
      // file extension.
      DatasetUtilities.downloadResource(
        filename: "\(subDirectory)", fileExtension: "zip",
        remoteRoot: url.deletingLastPathComponent(),
        localStorageDirectory: directory)
    #endif

    // Load the data files.
    let dataFilesURL = extractedDirectoryURL.appendingPathComponent("CoLA")
    trainingExamples = try CoLA.load(
      fromFile: dataFilesURL.appendingPathComponent("train.tsv")
    ).lazy.map(exampleMap)
    
    validationExamples = try CoLA.load(
      fromFile: dataFilesURL.appendingPathComponent("dev.tsv")
    ).lazy.map(exampleMap)

    self.maxSequenceLength = maxSequenceLength
    self.batchSize = batchSize

    // Create the training sequence of epochs.
    trainingEpochs = TrainingEpochs(
      samples: trainingExamples, batchSize: batchSize / maxSequenceLength, entropy: entropy
    ).lazy.map { (batches: Batches) -> LazyMapSequence<Batches, LabeledTextBatch> in
      batches.lazy.map{ 
        (
          data: $0.map(\.data).paddedAndCollated(to: maxSequenceLength),
          label: Tensor($0.map(\.label))
        )
      }
    }
    
    // Create the validation collection of batches.
    validationBatches = validationExamples.inBatches(of: batchSize / maxSequenceLength).lazy.map{ 
      (
        data: $0.map(\.data).paddedAndCollated(to: maxSequenceLength),
        label: Tensor($0.map(\.label))
      )
    }
  }
}

extension CoLA where Entropy == SystemRandomNumberGenerator {
  /// Creates an instance in `taskDirectoryURL` with batches of size `batchSize`
  /// by `maximumSequenceLength`.
  ///
  /// - Parameter exampleMap: a transform that processes `Example` in `LabeledTextBatch`.
  public init(
    taskDirectoryURL: URL,
    maxSequenceLength: Int,
    batchSize: Int,
    exampleMap: @escaping (CoLAExample) -> LabeledTextBatch
  ) throws {
    try self.init(
      taskDirectoryURL: taskDirectoryURL,
      maxSequenceLength: maxSequenceLength,
      batchSize: batchSize,
      entropy: SystemRandomNumberGenerator(),
      exampleMap: exampleMap
    )
  }
}