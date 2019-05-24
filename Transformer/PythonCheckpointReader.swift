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

import TensorFlow

struct Config: Codable {
    let vocabSize: Int
    let contextSize: Int
    let embeddingSize: Int
    let headCount: Int
    let layerCount: Int

    enum CodingKeys: String, CodingKey {
        case vocabSize = "n_vocab"
        case contextSize = "n_ctx"
        case embeddingSize = "n_embd"
        case headCount = "n_head"
        case layerCount = "n_layer"
    }
}

func readTensor<Scalar: TensorFlowScalar>(
    fromPath path: String,
    name: String,
    scalarType: Scalar.Type
) -> Tensor<Scalar> {
    // TODO(jekbradbury): support variadic dtype attrs in RawOpsGenerated
    return Raw.restoreV2(prefix: StringTensor(path),
                         tensorNames: StringTensor([name]),
                         shapeAndSlices: StringTensor([""]))
}

protocol InitializableFromPythonCheckpoint {
    init(contentsOfPythonCheckpointFile path: String, config: Config, scope: String)
}

extension Dense: InitializableFromPythonCheckpoint {
    init(contentsOfPythonCheckpointFile path: String, config: Config, scope: String) {
        let kernel = readTensor(fromPath: path, name: scope + "/w", scalarType: Scalar.self)
        self.init(
            weight: kernel.squeezingShape(at: 0),
            bias: readTensor(fromPath: path, name: scope + "/b", scalarType: Scalar.self),
            activation: identity)
    }
    init(
        contentsOfPythonCheckpointFile path: String,
        config: Config,
        scope: String,
        activation: String
    ) {
        let kernel = readTensor(fromPath: path, name: scope + "/w", scalarType: Scalar.self)
        self.init(
            weight: kernel.squeezingShape(at: 0),
            bias: readTensor(fromPath: path, name: scope + "/b", scalarType: Scalar.self),
            activation: gelu)
    }
}

extension LayerNorm: InitializableFromPythonCheckpoint {
    init(contentsOfPythonCheckpointFile path: String, config: Config, scope: String) {
        self.init(
            offset: readTensor(fromPath: path, name: scope + "/b", scalarType: Scalar.self),
            scale: readTensor(fromPath: path, name: scope + "/g", scalarType: Scalar.self),
            axis: -1,
            epsilon: Tensor(1e-5))
    }
}

extension MultiHeadAttention: InitializableFromPythonCheckpoint {
    init(contentsOfPythonCheckpointFile path: String, config: Config, scope: String) {
        attention = Attention(
            size: config.embeddingSize / config.headCount,
            causal: true,
            dropProbability: 0.2)
        wqkv = TimeDistributed(Dense<Float>(
            contentsOfPythonCheckpointFile: path,
            config: config,
            scope: scope + "/c_attn"))
        wo = TimeDistributed(Dense<Float>(
            contentsOfPythonCheckpointFile: path,
            config: config,
            scope: scope + "/c_proj"))
        headCount = config.headCount
    }
}

extension FeedForward: InitializableFromPythonCheckpoint {
    init(contentsOfPythonCheckpointFile path: String, config: Config, scope: String) {
        dense1 = TimeDistributed(Dense<Float>(
            contentsOfPythonCheckpointFile: path,
            config: config,
            scope: scope + "/c_fc",
            activation: "gelu"))
        dense2 = TimeDistributed(Dense<Float>(
            contentsOfPythonCheckpointFile: path,
            config: config,
            scope: scope + "/c_proj"))
        dropout = Dropout(probability: 0.2)
    }
}

extension EncoderLayer: InitializableFromPythonCheckpoint {
    init(contentsOfPythonCheckpointFile path: String, config: Config, scope: String) {
        selfAttention = MultiHeadAttention(
            contentsOfPythonCheckpointFile: path, config: config, scope: scope + "/attn")
        selfAttentionDropout = Dropout(probability: 0.2)
        selfAttentionNorm = LayerNorm(
            contentsOfPythonCheckpointFile: path, config: config, scope: scope + "/ln_1")
        feedForward = FeedForward(
            contentsOfPythonCheckpointFile: path, config: config, scope: scope + "/mlp")
        feedForwardDropout = Dropout(probability: 0.2)
        feedForwardNorm = LayerNorm(
            contentsOfPythonCheckpointFile: path, config: config, scope: scope + "/ln_2")
    }
}

extension TransformerLM: InitializableFromPythonCheckpoint {
    init(contentsOfPythonCheckpointFile path: String, config: Config, scope: String) {
        embedding = Embedding(
            weight: readTensor(fromPath: path, name: scope + "/wte", scalarType: Float.self))
        positionalEmbeddings = readTensor(
            fromPath: path,
            name: scope + "/wpe",
            scalarType: Float.self)
        layers = (0..<config.layerCount).map { i in
            EncoderLayer(
                contentsOfPythonCheckpointFile: path, config: config, scope: scope + "/h\(i)")
        }
        norm = LayerNorm(
            contentsOfPythonCheckpointFile: path, config: config, scope: scope + "/ln_f")
    }
}
