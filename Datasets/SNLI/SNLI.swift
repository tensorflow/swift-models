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
import TensorFlow

/// A `TextBatch` with the corresponding labels.
public typealias LabeledTextBatch = (data: TextBatch, label: Tensor<Int32>)

/// SNLI example.
public struct SNLIExample {
  /// The unique identifier representing the `Example`.
  public let id: String
  /// The premise
  public let premise: String
  // The hypothesis 
  public let hypothesis: String
  /// The label of the `Example`.
  public let label: String

  /// Creates an instance from `id`, `sentence` and `isAcceptable`.
  public init(id: String, premise: String, hypothesis: String, isAcceptable: Bool?) {
    self.id = id
    self.premise = premise
    self.hypothesis = hypothesis
    self.label = label
  }
}

public struct SNLI<Entropy: RandomNumberGenerator> {
  /// The directory where the dataset will be downloaded
  public let directoryURL: URL
  /// The type of the labeled samples.
  public typealias Samples = LazyMapSequence<[CoLAExample], LabeledTextBatch>
  /// The training texts.
  public let trainingExamples: Samples
  /// The validation texts.
  public let validationExamples: Samples
    
  /// The sequence length to which every sentence will be padded.
  public let maxSequenceLength: Int
  /// The batch size.
  public let batchSize: Int
    
  /// The type of the collection of batches.
  public typealias Batches = Slices<Sampling<Samples, ArraySlice<Int>>>
  /// The type of the training sequence of epochs.
  public typealias TrainEpochs = LazyMapSequence<TrainingEpochs<Samples, Entropy>, 
    LazyMapSequence<Batches, LabeledTextBatch>>
  /// The sequence of training data (epochs of batches).
  public var trainingEpochs: TrainEpochs
  /// The validation batches.
  public var validationBatches: LazyMapSequence<Slices<Samples>, LabeledTextBatch>
    
  /// The url from which to download the dataset.
  private let url: URL = URL(
    string: String("https://nlp.stanford.edu/projects/snli/snli_1.0.zip"))!
}

// Data
extension SNLI {
  internal static func load(fromFile fileURL: URL, isTest: Bool = false) throws -> [SNLIExample] {
    let lines = try parse(tsvFileAt: fileURL)

    if isTest {
      // The test data file has a header.
      return lines.dropFirst().enumerated().map { (i, lineParts) in
        SNLIExample(id: lineParts[8], 
                    premise: lineParts[5], 
                    hypothesis: lineparts[6], 
                    label: lineParts[0])
      }
    }

    return lines.dropFirst.enumerated().map { (i, lineParts) in
      SNLIExample(id: lineParts[8], 
                    premise: lineParts[5], 
                    hypothesis: lineparts[6], 
                    label: lineParts[0])
      }
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

extension SNLI {
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
    self.directoryURL = taskDirectoryURL.appendingPathComponent("SNLI")
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
    let dataFilesURL = extractedDirectoryURL.appendingPathComponent("SNLI")
    trainingExamples = try CoLA.load(
      fromFile: dataFilesURL.appendingPathComponent("snli_1.0_train.txt")
    ).lazy.map(exampleMap)
    
    validationExamples = try CoLA.load(
      fromFile: dataFilesURL.appendingPathComponent("snli_1.0_dev.txt")
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

extension SNLI where Entropy == SystemRandomNumberGenerator {
  /// Creates an instance in `taskDirectoryURL` with batches of size `batchSize`
  /// by `maximumSequenceLength`.
  ///
  /// - Parameter exampleMap: a transform that processes `Example` in `LabeledTextBatch`.
  public init(
    taskDirectoryURL: URL,
    maxSequenceLength: Int,
    batchSize: Int,
    exampleMap: @escaping (SNLIExample) -> LabeledTextBatch
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