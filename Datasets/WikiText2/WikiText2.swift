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

// Original source:
// "WikiText-2"
// Einstein AI
// https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/

import Batcher
import Foundation
import ModelSupport
import TensorFlow

public struct WikiText2 {
    public let trainingDataset: LanguageModelDataset<[Int]>
    public let validationDataset: LanguageModelDataset<[Int]>
    let bpe: BytePairEncoder

    public init(bpe: BytePairEncoder) {
        self.init(
            localStorageDirectory: FileManager.default.temporaryDirectory.appendingPathComponent(
            "WikiText2", isDirectory: true), bpe: bpe
    ) }

    public init(localStorageDirectory: URL, bpe: BytePairEncoder) {
        do {
            self.trainingDataset = try loadWikiText2Training(
                    localStorageDirectory: localStorageDirectory, bpe: bpe)
            self.validationDataset = try loadWikiText2Validation(
                    localStorageDirectory: localStorageDirectory, bpe: bpe)
            self.bpe = bpe
        } catch {
            fatalError("Could not load WikiText2 dataset: \(error)")
        }
    }
}

func downloadWikiText2IfNotPresent(to directory: URL) {
    let downloadPath = directory.appendingPathComponent("wikitext-2").path
    let directoryExists = FileManager.default.fileExists(atPath: downloadPath)
    let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: downloadPath)
    let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)

    guard !directoryExists || directoryEmpty else { return }

    let location = URL(
        string: "https://s3.amazonaws.com/fast-ai-nlp/")!
    let _ = DatasetUtilities.downloadResource(
        filename: "wikitext-2", fileExtension: "tgz",
        remoteRoot: location, localStorageDirectory: directory)
}

func readCSV(in file: URL) throws -> [String] {
    let rawText = try! String(contentsOf: file, encoding: .utf8)
    var rows = rawText.components(separatedBy: "\"\n\"")
    //Removing the initial "
    rows[0] = String(rows[0].dropFirst())
    //Removing the last "\n
    rows[rows.count-1] = String(rows[rows.count-1].substring(to: rows[rows.count-1].count-2))
    return rows
}

func embedding(for string: String, bpe: BytePairEncoder) -> [Int] {
    let tokens = bpe.encode(token: string, variant: .gpt2)
    // TODO(michellecasbon): Decide how to prevent OOV or choose a better ID (probably not 0).
    let ids = tokens.map { bpe.vocabulary.id(forToken: $0) ?? 0 }
    return ids
}

func loadWikiText2Directory(
    named name: String, in directory: URL, bpe: BytePairEncoder) throws -> LanguageModelDataset<[Int]> {
    downloadWikiText2IfNotPresent(to: directory)
    let path = directory.appendingPathComponent("wikitext-2/\(name).csv")

    let documentsFull = try readCSV(in: path)
    // TODO(michellecasbon): Process a larger number of documents.
    let documents = Array(documentsFull[0..<1])

    let embeddings = documents.map{ embedding(for: $0, bpe: bpe) }
    let lengths = embeddings.map{ $0.count }

    return LanguageModelDataset(
        batchSize: 4,
        sequenceLength: 64,
        items: embeddings,
        lengths: lengths
    )
}

func loadWikiText2Training(localStorageDirectory: URL, bpe: BytePairEncoder) throws
    -> LanguageModelDataset<[Int]>
{
    return try loadWikiText2Directory(
        named: "train", in: localStorageDirectory, bpe: bpe)
}

func loadWikiText2Validation(localStorageDirectory: URL, bpe: BytePairEncoder) throws
    -> LanguageModelDataset<[Int]>
{
    return try loadWikiText2Directory(
        named: "test", in: localStorageDirectory, bpe: bpe)
}

