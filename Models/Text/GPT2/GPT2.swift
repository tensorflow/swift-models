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

public class GPT2 {
  private let checkpoint: URL =
      URL(string: "https://storage.googleapis.com/gpt-2/models/117M/model.ckpt")!
  private let auxiliary: [String] = [
    "checkpoint",
    "encoder.json",
    "hparams.json",
    "model.ckpt.meta",
    "vocab.bpe",
  ]

  private let FS: FileManager = FileManager.default

  enum GPT2Error: Error {
  case invalidEncoding(id: Int32)
  }

  let storage: URL
  let configuration: (file: URL, data: Data)
  let encoder: (file: URL, data: Data)

  let reader: CheckpointReader
  let parameters: TransformerLMConfig
  // public let model: TransformerLM
  public let model: TransformerGPT2
  public let bpe: BytePairEncoder
  let mapping: BijectiveDictionary<String, Int32>

  public var seed: Tensor<Int32>
  public var temperature: Float = 1.0

  var states: [AttentionContext]

  public init() throws {
    storage = FS.temporaryDirectory.appendingPathComponent("Transformer")

    reader = try CheckpointReader(checkpointLocation: checkpoint,
                                  modelName: "Transformer",
                                  additionalFiles: auxiliary)

    // Load model from configuration
    let hparams_json: URL = storage.appendingPathComponent("hparams.json")
    configuration = try (hparams_json, Data(contentsOf: hparams_json))

    parameters = try JSONDecoder().decode(TransformerLMConfig.self,
                                          from: configuration.data)
    // model = TransformerLM(reader: reader, config: parameters, scope: "model")
    model = TransformerGPT2(reader: reader, config: parameters, scope: "model")

    // Load existing token mappings
    let encoder_json: URL = storage.appendingPathComponent("encoder.json")
    encoder = try (encoder_json, Data(contentsOf: encoder_json))

    let decoder: JSONDecoder = JSONDecoder()
    mapping =
        try BijectiveDictionary<String, Int32>(decoder.decode([String:Int32].self,
                                                              from: encoder.data))

    // Create bytepair encoder with loaded token mappings
    bpe = try BytePairEncoder(fromFileURL: encoder.file)

    // ...
    seed = Tensor(shape: [1, 1], scalars: [mapping["<|endoftext|>"]!])
    let empty: Tensor<Float> =
        Tensor<Float>(zeros: [parameters.headCount, 0,
                              parameters.embeddingSize / parameters.headCount])
    states = (0 ..< parameters.layerCount).map { _ in
      AttentionContext(key: empty, value: empty)
    }
  }

  public func embedding(for string: String) -> Tensor<Int32> {
    let tokens: [String] = bpe.encode(token: string, variant: .gpt2)
    // TODO(michellecasbon): Decide how to prevent OOV or choose a better ID (probably not 0).
    let ids: [Int32] = tokens.map { mapping[$0] ?? 0 }
    return Tensor(shape: [1, ids.count], scalars: ids)
  }

  public func generate() throws -> String {
//     let result = model(seed, states: &states)
    let result = model(seed)

    let (batchSize, timesteps, vocabularySize) =
        (result.shape[0], result.shape[1], result.shape[2])
    let logits =
        result.slice(lowerBounds: [0, timesteps - 1, 0],
                     upperBounds: [batchSize, timesteps, vocabularySize]) / temperature
    seed = Tensor(randomCategorialLogits: logits.squeezingShape(at: 1),
                  sampleCount: 1)

    let id: Int32 = Int32(seed[0][0])!
    if let token: String = mapping.key(id) {
      return BytePairEncoder.decode(token: token)
    }

    throw GPT2Error.invalidEncoding(id: id)
  }
}

