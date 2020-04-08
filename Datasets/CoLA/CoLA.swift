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

public struct CoLA {
    public let directoryURL: URL
    public let trainExamples: [Example]
    public let devExamples: [Example]
    public let testExamples: [Example]
    public let maxSequenceLength: Int
    public let batchSize: Int

    public typealias ExampleIterator = IndexingIterator<[Example]>
    public typealias TrainDataIterator = PrefetchIterator<
        GroupedIterator<MapIterator<ExampleIterator, DataBatch>>
    >
    public typealias DevDataIterator = GroupedIterator<MapIterator<ExampleIterator, DataBatch>>
    public typealias TestDataIterator = DevDataIterator

    public var trainDataIterator: TrainDataIterator
    public var devDataIterator: DevDataIterator
    public var testDataIterator: TestDataIterator
}

//===-----------------------------------------------------------------------------------------===//
// Data
//===-----------------------------------------------------------------------------------------===//

extension CoLA {
    /// CoLA example.
    public struct Example {
        public let id: String
        public let sentence: String
        public let isAcceptable: Bool?

        public init(id: String, sentence: String, isAcceptable: Bool?) {
            self.id = id
            self.sentence = sentence
            self.isAcceptable = isAcceptable
        }
    }

    /// CoLA data batch.
    public struct DataBatch: KeyPathIterable {
        public var inputs: TextBatch  // TODO: !!! Mutable in order to allow for batching.
        public var labels: Tensor<Int32>?  // TODO: !!! Mutable in order to allow for batching.

        public init(inputs: TextBatch, labels: Tensor<Int32>?) {
            self.inputs = inputs
            self.labels = labels
        }
    }

    /// URL pointing to the downloadable ZIP file that contains the CoLA dataset.
    private static let url: URL = URL(
        string: String(
            "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/"
                + "o/data%2FCoLA.zip?alt=media&token=46d5e637-3411-4188-bc44-5809b5bfb5f4"))!

    internal enum FileType: String {
        case train = "train"
        case dev = "dev"
        case test = "test"
    }

    internal static func load(fromFile fileURL: URL, fileType: FileType) throws -> [Example] {
        let lines = try parse(tsvFileAt: fileURL)

        if fileType == .test {
            // The test data file has a header.
            return lines.dropFirst().enumerated().map { (i, lineParts) in
                Example(id: lineParts[0], sentence: lineParts[1], isAcceptable: nil)
            }
        }

        return lines.enumerated().map { (i, lineParts) in
            Example(id: lineParts[0], sentence: lineParts[3], isAcceptable: lineParts[1] == "1")
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
    public init(
        exampleMap: @escaping (Example) -> DataBatch,
        taskDirectoryURL: URL,
        maxSequenceLength: Int,
        batchSize: Int,
        dropRemainder: Bool
    ) throws {
        self.directoryURL = taskDirectoryURL.appendingPathComponent("CoLA")
        let dataURL = directoryURL.appendingPathComponent("data")
        let compressedDataURL = dataURL.appendingPathComponent("downloaded-data.zip")

        // Download the data, if necessary.
        try download(from: CoLA.url, to: compressedDataURL)

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

        // Load the data files into arrays of examples.
        let dataFilesURL = extractedDirectoryURL.appendingPathComponent("CoLA")
        self.trainExamples = try CoLA.load(
            fromFile: dataFilesURL.appendingPathComponent("train.tsv"),
            fileType: .train)
        self.devExamples = try CoLA.load(
            fromFile: dataFilesURL.appendingPathComponent("dev.tsv"),
            fileType: .dev)
        self.testExamples = try CoLA.load(
            fromFile: dataFilesURL.appendingPathComponent("test.tsv"),
            fileType: .test)

        self.maxSequenceLength = maxSequenceLength
        self.batchSize = batchSize

        // Create the data iterators used for training and evaluating.
        self.trainDataIterator = trainExamples.shuffled().makeIterator()  // TODO: [RNG] Seed support.
            .map(exampleMap)
            .grouped(
                keyFn: { _ in 0 },
                sizeFn: { _ in batchSize / maxSequenceLength },
                reduceFn: {
                    DataBatch(
                        inputs: padAndBatch(
                            textBatches: $0.map { $0.inputs }, maxLength: maxSequenceLength),
                        labels: Tensor.batch($0.map { $0.labels! }))
                },
                dropRemainder: dropRemainder
            )
            .prefetched(count: 2)
        self.devDataIterator = devExamples.makeIterator()
            .map(exampleMap)
            .grouped(
                keyFn: { _ in 0 },
                sizeFn: { _ in batchSize / maxSequenceLength },
                reduceFn: {
                    DataBatch(
                        inputs: padAndBatch(
                            textBatches: $0.map { $0.inputs }, maxLength: maxSequenceLength),
                        labels: Tensor.batch($0.map { $0.labels! }))
                },
                dropRemainder: dropRemainder
            )
        self.testDataIterator = testExamples.makeIterator()
            .map(exampleMap)
            .grouped(
                keyFn: { _ in 0 },
                sizeFn: { _ in batchSize / maxSequenceLength },
                reduceFn: {
                    DataBatch(
                        inputs: padAndBatch(
                            textBatches: $0.map { $0.inputs }, maxLength: maxSequenceLength),
                        labels: nil)
                },
                dropRemainder : dropRemainder
            )
    }
}
