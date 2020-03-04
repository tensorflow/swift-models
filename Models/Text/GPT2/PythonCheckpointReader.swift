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

import ModelSupport
import TensorFlow

public struct TransformerLMConfig: Codable {
    public let vocabSize: Int
    public let contextSize: Int
    public let embeddingSize: Int
    public let headCount: Int
    public let layerCount: Int

    enum CodingKeys: String, CodingKey {
        case vocabSize = "n_vocab"
        case contextSize = "n_ctx"
        case embeddingSize = "n_embd"
        case headCount = "n_head"
        case layerCount = "n_layer"
    }
}

extension CheckpointReader {
    func readTensor<Scalar: TensorFlowScalar>(
        name: String,
        scalarType: Scalar.Type
    ) -> Tensor<Scalar> {
        return Tensor<Scalar>(loadTensor(named: name))
    }
}

protocol InitializableFromPythonCheckpoint {
    init(reader: CheckpointReader, config: TransformerLMConfig, scope: String)
}

extension Dense: InitializableFromPythonCheckpoint {
    init(reader: CheckpointReader, config: TransformerLMConfig, scope: String) {
        let kernel = reader.readTensor(name: scope + "/w", scalarType: Scalar.self)
        self.init(
            weight: kernel.squeezingShape(at: 0),
            bias: reader.readTensor(name: scope + "/b", scalarType: Scalar.self),
            activation: identity)
    }

    init(
        reader: CheckpointReader,
        config: TransformerLMConfig,
        scope: String,
        activation: String
    ) {
        let kernel = reader.readTensor(name: scope + "/w", scalarType: Scalar.self)
        self.init(
            weight: kernel.squeezingShape(at: 0),
            bias: reader.readTensor(name: scope + "/b", scalarType: Scalar.self),
            activation: gelu)
    }
}

extension LayerNorm: InitializableFromPythonCheckpoint {
    init(reader: CheckpointReader, config: TransformerLMConfig, scope: String) {
        self.init(
            offset: reader.readTensor(name: scope + "/b", scalarType: Scalar.self),
            scale: reader.readTensor(name: scope + "/g", scalarType: Scalar.self),
            axis: -1,
            epsilon: 1e-5)
    }
}

extension MultiHeadAttentionGPT2: InitializableFromPythonCheckpoint {
    init(reader: CheckpointReader, config: TransformerLMConfig, scope: String) {
        attention = Attention(
            size: config.embeddingSize / config.headCount,
            causal: true,
            dropProbability: 0.2)
        wqkv = TimeDistributed(
            Dense<Float>(reader: reader, config: config, scope: scope + "/c_attn"))
        wo = TimeDistributed(
            Dense<Float>(reader: reader, config: config, scope: scope + "/c_proj"))
        headCount = config.headCount
    }
}

extension FeedForward: InitializableFromPythonCheckpoint {
    init(reader: CheckpointReader, config: TransformerLMConfig, scope: String) {
        dense1 = TimeDistributed(
            Dense<Float>(reader: reader, config: config, scope: scope + "/c_fc", activation: "gelu")
        )
        dense2 = TimeDistributed(
            Dense<Float>(reader: reader, config: config, scope: scope + "/c_proj"))
        dropout = Dropout(probability: 0.2)
    }
}

extension EncoderLayer: InitializableFromPythonCheckpoint {
    init(reader: CheckpointReader, config: TransformerLMConfig, scope: String) {
        selfAttention = MultiHeadAttentionGPT2(reader: reader, config: config, scope: scope + "/attn")
        selfAttentionDropout = Dropout(probability: 0.2)
        selfAttentionNorm = LayerNorm(reader: reader, config: config, scope: scope + "/ln_1")
        feedForward = FeedForward(reader: reader, config: config, scope: scope + "/mlp")
        feedForwardDropout = Dropout(probability: 0.2)
        feedForwardNorm = LayerNorm(reader: reader, config: config, scope: scope + "/ln_2")
    }
}

extension TransformerLM: InitializableFromPythonCheckpoint {
    public init(reader: CheckpointReader, config: TransformerLMConfig, scope: String) {
        embedding = EmbeddingGPT2(
            weight: reader.readTensor(name: scope + "/wte", scalarType: Float.self))
        positionalEmbeddings = reader.readTensor(
            name: scope + "/wpe",
            scalarType: Float.self)
        layers = (0..<config.layerCount).map { i in
            EncoderLayer(reader: reader, config: config, scope: scope + "/h\(i)")
        }
        norm = LayerNorm(reader: reader, config: config, scope: scope + "/ln_f")
    }
}
