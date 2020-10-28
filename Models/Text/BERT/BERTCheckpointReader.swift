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

import Checkpoints
import Datasets
import Foundation
import ModelSupport
import TensorFlow

extension TransformerEncoderLayer {
  public mutating func load(bert reader: CheckpointReader, prefix: String) {
    multiHeadAttention.queryWeight = reader.readTensor(
      name: "\(prefix)/attention/self/query/kernel")
    multiHeadAttention.queryBias = reader.readTensor(name: "\(prefix)/attention/self/query/bias")
    multiHeadAttention.keyWeight = reader.readTensor(name: "\(prefix)/attention/self/key/kernel")
    multiHeadAttention.keyBias = reader.readTensor(name: "\(prefix)/attention/self/key/bias")
    multiHeadAttention.valueWeight = reader.readTensor(
      name: "\(prefix)/attention/self/value/kernel")
    multiHeadAttention.valueBias = reader.readTensor(name: "\(prefix)/attention/self/value/bias")
    attentionWeight = reader.readTensor(name: "\(prefix)/attention/output/dense/kernel")
    attentionBias = reader.readTensor(name: "\(prefix)/attention/output/dense/bias")
    attentionLayerNorm.offset = reader.readTensor(name: "\(prefix)/attention/output/LayerNorm/beta")
    attentionLayerNorm.scale = reader.readTensor(name: "\(prefix)/attention/output/LayerNorm/gamma")
    intermediateWeight = reader.readTensor(name: "\(prefix)/intermediate/dense/kernel")
    intermediateBias = reader.readTensor(name: "\(prefix)/intermediate/dense/bias")
    outputWeight = reader.readTensor(name: "\(prefix)/output/dense/kernel")
    outputBias = reader.readTensor(name: "\(prefix)/output/dense/bias")
    outputLayerNorm.offset = reader.readTensor(name: "\(prefix)/output/LayerNorm/beta")
    outputLayerNorm.scale = reader.readTensor(name: "\(prefix)/output/LayerNorm/gamma")
  }

  public mutating func load(albert reader: CheckpointReader, prefix: String) {
    multiHeadAttention.queryWeight = reader.readTensor(
      name: "\(prefix)/attention_1/self/query/kernel")
    multiHeadAttention.queryBias = reader.readTensor(name: "\(prefix)/attention_1/self/query/bias")
    multiHeadAttention.keyWeight = reader.readTensor(name: "\(prefix)/attention_1/self/key/kernel")
    multiHeadAttention.keyBias = reader.readTensor(name: "\(prefix)/attention_1/self/key/bias")
    multiHeadAttention.valueWeight = reader.readTensor(
      name: "\(prefix)/attention_1/self/value/kernel")
    multiHeadAttention.valueBias = reader.readTensor(name: "\(prefix)/attention_1/self/value/bias")
    attentionWeight = reader.readTensor(name: "\(prefix)/attention_1/output/dense/kernel")
    attentionBias = reader.readTensor(name: "\(prefix)/attention_1/output/dense/bias")
    attentionLayerNorm.offset = reader.readTensor(name: "\(prefix)/LayerNorm/beta")
    attentionLayerNorm.scale = reader.readTensor(name: "\(prefix)/LayerNorm/gamma")
    intermediateWeight = reader.readTensor(name: "\(prefix)/ffn_1/intermediate/dense/kernel")
    intermediateBias = reader.readTensor(name: "\(prefix)/ffn_1/intermediate/dense/bias")
    outputWeight = reader.readTensor(name: "\(prefix)/ffn_1/intermediate/output/dense/kernel")
    outputBias = reader.readTensor(name: "\(prefix)/ffn_1/intermediate/output/dense/bias")
    outputLayerNorm.offset = reader.readTensor(name: "\(prefix)/LayerNorm_1/beta")
    outputLayerNorm.scale = reader.readTensor(name: "\(prefix)/LayerNorm_1/gamma")
  }
}
