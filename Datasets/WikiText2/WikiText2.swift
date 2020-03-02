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
// "WIkiText-2"
// Einstein AI
// https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/

import Foundation
import ModelSupport
import TensorFlow
import Batcher

public struct WikiText2 {
    public let trainingDataset: LanguageModelDataset<[Int]>
    public let validationDataset: LanguageModelDataset<[Int]>
    // public let trainingExampleCount = 12894
    // public let validationExampleCount = 500

    public init() {
        self.init(
            localStorageDirectory: FileManager.default.temporaryDirectory.appendingPathComponent(
            "WikiText2", isDirectory: true)
    ) }

    public init(localStorageDirectory: URL) {
        do {
            self.trainingDataset = try loadWikiText2Training(
                    localStorageDirectory: localStorageDirectory)
            self.validationDataset = try loadWikiText2Validation(
                    localStorageDirectory: localStorageDirectory)
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

func easyTokenize(_ text: String) -> [String] {
    return text.components(separatedBy: " ")
}

func countTokens(_ texts: [[String]]) -> ([Int], [String:Int]) {
    var counts: [String:Int] = [:]
    var lengths: [Int] = []
    for tokens in texts {
        lengths.append(tokens.count)
        for token in tokens {
            counts[token] = (counts[token] ?? 0) + 1
        }
    }
    return (lengths,counts)
}

func makeVocabulary(
    _ counts: [String:Int], 
    minFrequency: Int = 2, 
    maxCount: Int = 60000) 
-> (itos: [Int:String], stoi: [String:Int]) {
    let withoutSpec = counts.filter { $0.0 != "xxunk" && $0.0 != "xxpad" }
    let sorted = withoutSpec.sorted { $0.1 > $1.1 }
    var itos: [Int:String] = [0:"xxunk", 1:"xxpad"]
    var stoi: [String:Int] = ["xxunk":0, "xxpad":1]
    for (i,x) in sorted.enumerated() {
        if i+2 >= maxCount || x.1 < minFrequency { break }
        itos[i+2] = (x.0)
        stoi[x.0] = i+2
    }
    return (itos: itos, stoi: stoi)
}

func numericalize(_ tokens: [String], with stoi: [String:Int]) -> [Int] {
    return tokens.map { stoi[$0] ?? 6 }
}

func loadWikiText2Directory(
    named name: String, in directory: URL) throws -> LanguageModelDataset<[Int]> {
    downloadWikiText2IfNotPresent(to: directory)
    let path = directory.appendingPathComponent("wikitext-2/\(name).csv")

    let documents = try readCSV(in: path)

    // TODO(michellecasbon): Replace with BytePairEncoder.
    let documentsTokenized = documents.map(easyTokenize)
    let (lengths, counts) = countTokens(documentsTokenized)
    let vocabulary = makeVocabulary(counts)
    let numericalizedTexts = documentsTokenized.map{ numericalize($0, with: vocabulary.stoi) }

    return LanguageModelDataset(
        batchSize: 64, 
        sequenceLength: 72, 
        items: numericalizedTexts, 
        lengths: lengths
    )
}

func loadWikiText2Training(localStorageDirectory: URL) throws
    -> LanguageModelDataset<[Int]>
{
    return try loadWikiText2Directory(
        named: "train", in: localStorageDirectory)
}

func loadWikiText2Validation(localStorageDirectory: URL) throws
    -> LanguageModelDataset<[Int]>
{
    return try loadWikiText2Directory(
        named: "test", in: localStorageDirectory)
}

