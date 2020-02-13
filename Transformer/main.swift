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
import Python
import TensorFlow

let modelName = "117M"

let remoteCheckpoint = URL(
    string: "https://storage.googleapis.com/gpt-2/models/\(modelName)/model.ckpt")!

let reader = try CheckpointReader(
    checkpointLocation: remoteCheckpoint, modelName: "Transformer",
    additionalFiles: ["checkpoint", "encoder.json", "hparams.json", "model.ckpt.meta", "vocab.bpe"])

let temporaryDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(
    "Transformer")
let sys = Python.import("sys")
sys.path = sys.path + ["./Transformer"]
let encoder = Python.import("encoder").get_encoder(temporaryDirectory.path)

let configFile = temporaryDirectory.appendingPathComponent("hparams.json")
let configData = try Data(contentsOf: configFile)
let config = try JSONDecoder().decode(TransformerLMConfig.self, from: configData)
let model = TransformerLM(reader: reader, config: config, scope: "model")

let start_token = Int32(encoder.encoder["<|endoftext|>"])!
var tokens = Tensor(shape: [1, 1], scalars: [start_token])
var temperature = Float(1.0)

if CommandLine.arguments.count >= 2 {
    temperature = Float(CommandLine.arguments[1])!
}

if CommandLine.arguments.count == 3 {
    let seed = CommandLine.arguments[2]
    print(seed, terminator: "")
    let pytok = encoder.encode(seed)
    let tokarr: [Int32] = [Int](pytok)!.map { Int32($0) }
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
    print(encoder.decode(tokens[0].makeNumpyArray()), terminator: "")
}
print()
