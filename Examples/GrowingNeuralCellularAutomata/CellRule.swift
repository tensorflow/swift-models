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

struct CellRule: Layer {
  @noDerivative var horizontalSobelFilter: Tensor<Float>
  @noDerivative var verticalSobelFilter: Tensor<Float>
  @noDerivative let fireRate: Float

  var conv1: Conv2D<Float>
  var conv2: Conv2D<Float>
  
  init(stateChannels: Int, fireRate: Float) {
    self.fireRate = fireRate
    
    let horizontalSobelKernel = Tensor<Float>(shape: [3, 3, 1, 1], scalars: [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0])
    horizontalSobelFilter = horizontalSobelKernel.broadcasted(to: [3, 3, stateChannels, 1])
    let verticalSobelKernel = Tensor<Float>(shape: [3, 3, 1, 1], scalars: [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0])
    verticalSobelFilter = verticalSobelKernel.broadcasted(to: [3, 3, stateChannels, 1])

    conv1 = Conv2D<Float>(filterShape: (1, 1, stateChannels * 3, 128))
    conv2 = Conv2D<Float>(filterShape: (1, 1, 128, stateChannels), filterInitializer: zeros())
  }

  @differentiable
  func livingMask(_ input: Tensor<Float>) -> Tensor<Float> {
    let alphaChannel = input.slice(lowerBounds: [0, 0, 0, 3], sizes: [input.shape[0], input.shape[1], input.shape[2], 1])
    let offset = maxPool2D(alphaChannel, filterSize: (1, 1, 1, 1), strides: (1, 1, 1, 1), padding: .same) - 0.1
    return max(0.0, sign(offset))
  }
  
  @differentiable
  func perceive(_ input: Tensor<Float>) -> Tensor<Float> {
    let horizontalSobel = depthwiseConv2D(input, filter: horizontalSobelFilter, strides: (1, 1, 1, 1), padding: .same)
    let verticalSobel = depthwiseConv2D(input, filter: verticalSobelFilter, strides: (1, 1, 1, 1), padding: .same)
    return Tensor(concatenating: [horizontalSobel, verticalSobel, input], alongAxis: 3)
  }

  @differentiable
  func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    let livingMaskBefore = livingMask(input)

    let perception = perceive(input)
    let dx = conv2(relu(conv1(perception)))
    
    let updateFireRate = Tensor<Float>(randomUniform: [input.shape[0], input.shape[1], input.shape[2], 1]) - fireRate
    let updateMask = max(0.0, sign(updateFireRate))

    let livingMaskAfter = livingMask(input)
    let combinedLivingMask = max(livingMaskBefore, livingMaskAfter)
    
    return (input + (dx * updateMask)) * combinedLivingMask
  }
}
