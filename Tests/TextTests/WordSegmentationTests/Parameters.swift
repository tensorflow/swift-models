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

import TensorFlow

struct SNLMParameters {
  var emb_enc: EmbeddingParameters
  var lstm_enc: LSTMParameters
  var mlp_interpolation: MLPParameters
  var mlp_memory: MLPParameters
  var emb_dec: EmbeddingParameters
  var lstm_dec: LSTMParameters
  var linear_dec: LinearParameters
}

struct EmbeddingParameters {
  var weight: Tensor<Float>
}

struct LSTMParameters {
  var weight_ih_l0: Tensor<Float>
  var weight_hh_l0: Tensor<Float>
  var bias_ih_l0: Tensor<Float>
  var bias_hh_l0: Tensor<Float>
}

struct MLPParameters {
  var linear1: LinearParameters
  var linear2: LinearParameters
}

struct LinearParameters {
  var weight: Tensor<Float>
  var bias: Tensor<Float>
}
