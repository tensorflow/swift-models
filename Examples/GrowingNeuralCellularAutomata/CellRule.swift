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
  @noDerivative var perceptionFilter: Tensor<Float>
  @noDerivative let fireRate: Float

  var conv1: Conv2D<Float>
  var conv2: Conv2D<Float>

  init(stateChannels: Int, fireRate: Float, useBias: Bool) {
    self.fireRate = fireRate

    let horizontalSobelKernel =
      Tensor<Float>(
        shape: [3, 3, 1, 1], scalars: [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0]) / 8.0
    let horizontalSobelFilter = horizontalSobelKernel.broadcasted(to: [3, 3, stateChannels, 1])
    let verticalSobelKernel =
      Tensor<Float>(
        shape: [3, 3, 1, 1], scalars: [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0]) / 8.0
    let verticalSobelFilter = verticalSobelKernel.broadcasted(to: [3, 3, stateChannels, 1])
    let identityKernel = Tensor<Float>(
      shape: [3, 3, 1, 1], scalars: [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    let identityFilter = identityKernel.broadcasted(to: [3, 3, stateChannels, 1])
    perceptionFilter = Tensor(
      concatenating: [horizontalSobelFilter, verticalSobelFilter, identityFilter], alongAxis: 3)

    conv1 = Conv2D<Float>(filterShape: (1, 1, stateChannels * 3, 128))
    conv2 = Conv2D<Float>(
      filterShape: (1, 1, 128, stateChannels), useBias: useBias, filterInitializer: zeros())
  }

  @differentiable
  func livingMask(_ input: Tensor<Float>) -> Tensor<Float> {
    let alphaChannel = input.slice(
      lowerBounds: [0, 0, 0, 3], sizes: [input.shape[0], input.shape[1], input.shape[2], 1])
    let localMaximum =
      maxPool2D(alphaChannel, filterSize: (1, 3, 3, 1), strides: (1, 1, 1, 1), padding: .same)
    return withoutDerivative(at: input) { _ in localMaximum.mask { $0 .> 0.1 } }
  }

  @differentiable
  func perceive(_ input: Tensor<Float>) -> Tensor<Float> {
    return depthwiseConv2D(
      input, filter: perceptionFilter, strides: (1, 1, 1, 1), padding: .same)
  }

  @differentiable
  func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    let livingMaskBefore = livingMask(input)

    let perception = perceive(input)
    let dx = conv2(relu(conv1(perception)))

    let updateDistribution =
      Tensor<Float>(
        randomUniform: [input.shape[0], input.shape[1], input.shape[2], 1], on: input.device)
    let updateMask = withoutDerivative(at: input) { _ in updateDistribution.mask { $0 .< fireRate }
    }

    let updatedState = input + (dx * updateMask)
    let livingMaskAfter = livingMask(updatedState)
    let combinedLivingMask = livingMaskBefore * livingMaskAfter

    return updatedState * combinedLivingMask
  }
}

func normalizeGradient(_ gradient: CellRule.TangentVector) -> CellRule.TangentVector {
  var outputGradient = gradient
  for kp in gradient.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
    let norm = sqrt(gradient[keyPath: kp].squared().sum())
    outputGradient[keyPath: kp] = gradient[keyPath: kp] / (norm + 1e-8)
  }

  return outputGradient
}

extension Tensor where Scalar: Numeric {
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  var colorComponents: Tensor {
    precondition(self.rank == 3 || self.rank == 4)
    if self.rank == 3 {
      return self.slice(
        lowerBounds: [0, 0, 0], sizes: [self.shape[0], self.shape[1], 4])
    } else {
      return self.slice(
        lowerBounds: [0, 0, 0, 0], sizes: [self.shape[0], self.shape[1], self.shape[2], 4])
    }
  }

  func mask(condition: (Tensor) -> Tensor<Bool>) -> Tensor {
    let satisfied = condition(self)
    return Tensor(zerosLike: self)
      .replacing(with: Tensor(onesLike: self), where: satisfied)
  }
}

// Note: the following is an identity function that serves to cut the backward trace into
// smaller identical traces, to improve X10 performance.
@inlinable
@differentiable
func clipBackwardsTrace(_ input: Tensor<Float>) -> Tensor<Float> {
  return input
}

@inlinable
@derivative(of: clipBackwardsTrace)
func _vjpClipBackwardsTrace(
  _ input: Tensor<Float>
) -> (value: Tensor<Float>, pullback: (Tensor<Float>) -> Tensor<Float>) {
  return (
    input,
    {
      LazyTensorBarrier()
      return $0
    }
  )
}
