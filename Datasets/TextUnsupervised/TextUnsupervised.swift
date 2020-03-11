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
    var fileExtension: String { get set }
}

public struct TextUnsupervised {
    private struct WikiText103Details: TextUnsupervisedVariantDetails {
        var variant = TextUnsupervisedVariant.wikiText103
        var location = URL(string: "https://s3.amazonaws.com/fast-ai-nlp/")!
        var trainingDirectoryName = "train"
        var validationDirectoryName = "test"
        var filename = "wikitext-103"
        var fileExtension = "tgz"
    }

    private struct WikiText2Details: TextUnsupervisedVariantDetails {
        var variant = TextUnsupervisedVariant.wikiText2
        var location = URL(string: "https://s3.amazonaws.com/fast-ai-nlp/")!
        var trainingDirectoryName = "train"
        var validationDirectoryName = "test"
        var filename = "wikitext-2"
        var fileExtension = "tgz"
    }

    public let trainingDataset: LanguageModelDataset<[Int]>
    public let validationDataset: LanguageModelDataset<[Int]>
    public let bpe: BytePairEncoder
    public let variant: TextUnsupervisedVariant
    private let variantDetails: TextUnsupervisedVariantDetails

    public init(
        variant: TextUnsupervisedVariant = TextUnsupervisedVariant.wikiText2
    ) {
        // Empty BytePairEncoder.
        let vocabulary = Vocabulary(tokensToIds: ["<|endoftext|>": 0])
        let mergePairs = [BytePairEncoder.Pair: Int]()
        let bpe = BytePairEncoder(vocabulary: vocabulary, mergePairs: mergePairs)

        self.init(bpe: bpe, variant: variant)
    }

    public init(
        bpe: BytePairEncoder,
        variant: TextUnsupervisedVariant = TextUnsupervisedVariant.wikiText2
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
                variantDetails: variantDetails)
            self.validationDataset = try TextUnsupervised.loadValidation(
                localStorageDirectory: localStorageDirectory, bpe: bpe,
                variantDetails: variantDetails)
        } catch {
            fatalError("Could not load dataset for \(variant): \(error)")
        }
    }

    private static func downloadIfNotPresent(
        to directory: URL, variantDetails: TextUnsupervisedVariantDetails
    ) {
        let downloadPath = directory.appendingPathComponent(variantDetails.variant.rawValue).path
        let directoryExists = FileManager.default.fileExists(atPath: downloadPath)
        let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: downloadPath)
        let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)

        guard !directoryExists || directoryEmpty else { return }

        // Downloads and extracts dataset files.
        let _ = DatasetUtilities.downloadResource(
            filename: variantDetails.filename, fileExtension: variantDetails.fileExtension,
            remoteRoot: variantDetails.location, localStorageDirectory: directory, extract: true)
    }

    private static func readCSV(in file: URL) throws -> [String] {
        let rawText = try! String(contentsOf: file, encoding: .utf8)
        var rows = rawText.components(separatedBy: "\"\n\"")
        // Removing the initial '"'.
        rows[0] = String(rows[0].dropFirst())
        // Removing the last '"\n'.
        rows[rows.count - 1] = String(
            rows[rows.count - 1].substring(to: rows[rows.count - 1].count - 2))
        return rows
    }

    private static func embedding(for string: String, bpe: BytePairEncoder) -> [Int] {
        let tokens = bpe.encode(token: string, variant: .gpt2)
        // TODO(michellecasbon): Decide how to prevent OOV or choose a better ID (probably not 0).
        let ids = tokens.map { bpe.vocabulary.id(forToken: $0) ?? 0 }
        return ids
    }

    // Only supports CSV files.
    private static func loadDirectory(
        named name: String, in directory: URL,
        bpe: BytePairEncoder, variantDetails: TextUnsupervisedVariantDetails
    ) throws -> LanguageModelDataset<[Int]> {
        downloadIfNotPresent(to: directory, variantDetails: variantDetails)
        let path = directory.appendingPathComponent("\(variantDetails.filename)/\(name).csv")

        let documentsFull = try readCSV(in: path)
        // TODO(michellecasbon): Process a larger number of documents.
        let documents = Array(documentsFull[0..<1])

        let embeddings = documents.map { embedding(for: $0, bpe: bpe) }
        let lengths = embeddings.map { $0.count }

        return LanguageModelDataset(
            batchSize: 64,
            sequenceLength: 72,
            items: embeddings,
            lengths: lengths
        )
    }

    private static func loadTraining(
        localStorageDirectory: URL, bpe: BytePairEncoder,
        variantDetails: TextUnsupervisedVariantDetails
    )
        throws
        -> LanguageModelDataset<[Int]>
    {
        return try loadDirectory(
            named: variantDetails.trainingDirectoryName, in: localStorageDirectory, bpe: bpe,
            variantDetails: variantDetails)
    }

    private static func loadValidation(
        localStorageDirectory: URL, bpe: BytePairEncoder,
        variantDetails: TextUnsupervisedVariantDetails
    )
        throws
        -> LanguageModelDataset<[Int]>
    {
        return try loadDirectory(
            named: variantDetails.validationDirectoryName, in: localStorageDirectory, bpe: bpe,
            variantDetails: variantDetails)
    }
}
