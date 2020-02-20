// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
import TextModels

let modelName = "117M"

let remoteCheckpoint = URL(
    string: "https://storage.googleapis.com/gpt-2/models/\(modelName)/model.ckpt")!

let reader = try CheckpointReader(
    checkpointLocation: remoteCheckpoint, modelName: "Transformer",
    additionalFiles: ["checkpoint", "encoder.json", "hparams.json", "model.ckpt.meta", "vocab.bpe"])

let temporaryDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(
    "Transformer")

let vocabularyFile = temporaryDirectory.appendingPathComponent("vocab.bpe")
let vocabulary = try(Vocabulary(fromFile: vocabularyFile))
let mergePairs = [BytePairEncoder.Pair: Int](
    uniqueKeysWithValues:
       (try String(contentsOfFile: vocabularyFile.path, encoding: .utf8))
            .components(separatedBy: .newlines)
            .dropFirst()
            .enumerated()
            .compactMap { (index, line) -> (BytePairEncoder.Pair, Int)? in
                let lineParts = line.split(separator: " ")
                if lineParts.count < 2 { return nil }
                    return (
                       BytePairEncoder.Pair(
                           String(lineParts[0]),
                           String(lineParts[1])),
                       index)
            })
let bytePairEncoder = BytePairEncoder(vocabulary: vocabulary, mergePairs: mergePairs)

let configFile = temporaryDirectory.appendingPathComponent("hparams.json")
let configData = try Data(contentsOf: configFile)
let config = try JSONDecoder().decode(TransformerLMConfig.self, from: configData)
let model = TransformerLM(reader: reader, config: config, scope: "model")

let encoderFile = temporaryDirectory.appendingPathComponent("encoder.json")
let encoderData = try Data(contentsOf: encoderFile)
let tokenToID: [String: Int32] = try JSONDecoder().decode([String: Int32].self, from: encoderData)
let IDToToken = [Int32: String](uniqueKeysWithValues: tokenToID.map{ ($1, $0) })

let start_token = tokenToID["<|endoftext|>"]!
var tokens = Tensor(shape: [1, 1], scalars: [start_token])
var temperature = Float(1.0)

if CommandLine.arguments.count >= 2 {
    temperature = Float(CommandLine.arguments[1])!
}

if CommandLine.arguments.count == 3 {
    let seed = CommandLine.arguments[2]
    print(seed, terminator: "")
    let pytok = bytePairEncoder.encode(token: seed)
    let tokarr: [Int32] = pytok.map { tokenToID[$0]! }
    tokens = Tensor(shape: [1, tokarr.count], scalars: tokarr)
}

let empty = Tensor<Float>(zeros: [config.headCount, 0, config.embeddingSize / config.headCount])
var states = (0..<config.layerCount).map { _ in AttentionContext(key: empty, value: empty) }

for _ in 0..<100 {
    let logits = model(tokens, states: &states)
    let (batchSize, timeSteps, vocabSize) = (logits.shape[0], logits.shape[1], logits.shape[2])
    let lastLogit =
        logits.slice(
            lowerBounds: [0, timeSteps - 1, 0],
            upperBounds: [batchSize, timeSteps, vocabSize]) / temperature
    tokens = Tensor(randomCategorialLogits: lastLogit.squeezingShape(at: 1), sampleCount: 1)

    var decodedToken: String
    let ID: Int32 = Int32(tokens[0][0])!
    if let token: String = IDToToken[ID] {
        decodedToken = bytePairEncoder.decode(token: token)
    } else {
        decodedToken = "ID \(ID) not found."
    }
    print(decodedToken, terminator: "")
}
print()
