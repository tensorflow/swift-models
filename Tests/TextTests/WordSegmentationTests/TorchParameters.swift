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

struct TorchSNLMParameters {
  var emb_enc: TorchEmbeddingParameters
  var lstm_enc: TorchLSTMParameters
  var mlp_interpolation: TorchMLPParameters
  var mlp_memory: TorchMLPParameters
  var emb_dec: TorchEmbeddingParameters
  var lstm_dec: TorchLSTMParameters
  var linear_dec: TorchLinearParameters
}

struct TorchEmbeddingParameters {
  var weight: Tensor<Float>
}

struct TorchLSTMParameters {
  var weight_ih_l0: Tensor<Float>
  var weight_hh_l0: Tensor<Float>
  var bias_ih_l0: Tensor<Float>
  var bias_hh_l0: Tensor<Float>
}

struct TorchMLPParameters {
  var linear1: TorchLinearParameters
  var linear2: TorchLinearParameters
}

struct TorchLinearParameters {
  var weight: Tensor<Float>
  var bias: Tensor<Float>
}
