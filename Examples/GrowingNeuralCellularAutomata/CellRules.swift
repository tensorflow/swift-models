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

struct CellRules: Layer {
  var conv1: Conv2D<Float>
  var conv2: Conv2D<Float>
  
  init(stateChannels: Int) {
    conv1 = Conv2D<Float>(filterShape: (1, 1, 4, 128))
    
    // TODO: Initialize last layer with all zeros
    conv2 = Conv2D<Float>(filterShape: (1, 1, 128, stateChannels))
  }

  @differentiable
  func perceive(_ input: Tensor<Float>) -> Tensor<Float> {
    // TODO: Sobel kernels here
    return input
  }

  @differentiable
  func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    // TODO: Get input living mask
    // TODO: Get output living mask using fire rate
    let perception = perceive(input)
    // TODO: Apply living mask to output
    return conv2(relu(conv1(perception)))
  }
}
