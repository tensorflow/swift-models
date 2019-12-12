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

// Original Paper:
// "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
// Applications"
// Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko
// Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
// https://arxiv.org/abs/1704.04861

public struct ConvBlock: Layer {
  public var zeroPad = ZeroPadding2D<Float>(padding: ((0, 1), (0, 1)))
  public var conv: Conv2D<Float>
  public var batchNorm: BatchNorm<Float>

  public init(filterCount: Int, strides: (Int, Int)) {
    conv = Conv2D<Float>(
      filterShape: (3, 3, 3, filterCount),
      strides: strides,
      padding: .valid)
    batchNorm = BatchNorm<Float>(featureCount: filterCount)
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    let convolved = input.sequenced(through: zeroPad, conv, batchNorm)
    return relu6(convolved)
  }
}

public struct DepthwiseConvBlock: Layer {
  @noDerivative
  let depthMultiplier: Int
  @noDerivative
  let strides: (Int, Int)

  @noDerivative
  public let zeroPad = ZeroPadding2D<Float>(padding: ((0, 1), (0, 1)))
  public var dConv: DepthwiseConv2D<Float>
  public var batchNorm1: BatchNorm<Float>
  public var conv: Conv2D<Float>
  public var batchNorm2: BatchNorm<Float>

  public init(filterCount: Int, pointwiseFilterCount: Int, depthMultiplier: Int,
    strides: (Int, Int)) {
    if depthMultiplier < 1 {
      fatalError("Depth multiplier must be an integer greater than 0.")
    }
    self.depthMultiplier = depthMultiplier
    self.strides = strides

    dConv = DepthwiseConv2D<Float>(
      filterShape: (3, 3, filterCount, self.depthMultiplier),
      strides: strides,
      padding: strides == (1, 1) ? .same : .valid)
    batchNorm1 = BatchNorm<Float>(
      featureCount: filterCount * self.depthMultiplier)
    conv = Conv2D<Float>(
      filterShape: (1, 1, filterCount * self.depthMultiplier,
        pointwiseFilterCount),
      strides: (1, 1),
      padding: .same)
    batchNorm2 = BatchNorm<Float>(featureCount: pointwiseFilterCount)
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    var convolved1: Tensor<Float>
    if self.strides == (1, 1) {
      convolved1 = input.sequenced(through: dConv, batchNorm1)
    }
    else {
      convolved1 = input.sequenced(through: zeroPad, dConv, batchNorm1)
    }
    let convolved2 = relu6(convolved1)
    let convolved3 = relu6(convolved2.sequenced(through: conv, batchNorm2))
    return convolved3
  }
}

public struct MobileNetV1: Layer {
  @noDerivative
  let classCount: Int

  public var convBlock1: ConvBlock
  public var dConvBlock1: DepthwiseConvBlock
  public var dConvBlock2: DepthwiseConvBlock
  public var dConvBlock3: DepthwiseConvBlock
  public var dConvBlock4: DepthwiseConvBlock
  public var dConvBlock5: DepthwiseConvBlock
  public var dConvBlock6: DepthwiseConvBlock
  public var dConvBlock7: DepthwiseConvBlock
  public var dConvBlock8: DepthwiseConvBlock
  public var dConvBlock9: DepthwiseConvBlock
  public var dConvBlock10: DepthwiseConvBlock
  public var dConvBlock11: DepthwiseConvBlock
  public var dConvBlock12: DepthwiseConvBlock
  public var dConvBlock13: DepthwiseConvBlock
  public var avgPool = GlobalAvgPool2D<Float>()
  public var reshape = Reshape<Float>(shape: [1, 1, 1, 1024])
  public var dropoutLayer: Dropout<Float>
  public var convLast: Conv2D<Float>

  public init(classCount: Int, depthMultiplier: Int = 1,
    dropout: Double = 0.001) {
    self.classCount = classCount

    convBlock1 = ConvBlock(filterCount: 32, strides: (2, 2))
    dConvBlock1 = DepthwiseConvBlock(
      filterCount: 32,
      pointwiseFilterCount: 64,
      depthMultiplier: depthMultiplier,
      strides: (1, 1))
    dConvBlock2 = DepthwiseConvBlock(
      filterCount: 64,
      pointwiseFilterCount: 128,
      depthMultiplier: depthMultiplier,
      strides: (2, 2))
    dConvBlock3 = DepthwiseConvBlock(
      filterCount: 128,
      pointwiseFilterCount: 128,
      depthMultiplier: depthMultiplier,
      strides: (1, 1))
    dConvBlock4 = DepthwiseConvBlock(
      filterCount: 128,
      pointwiseFilterCount: 256,
      depthMultiplier: depthMultiplier,
      strides: (2, 2))
    dConvBlock5 = DepthwiseConvBlock(
      filterCount: 256,
      pointwiseFilterCount: 256,
      depthMultiplier: depthMultiplier,
      strides: (1, 1))
    dConvBlock6 = DepthwiseConvBlock(
      filterCount: 256,
      pointwiseFilterCount: 512,
      depthMultiplier: depthMultiplier,
      strides: (2, 2))
    dConvBlock7 = DepthwiseConvBlock(
      filterCount: 512,
      pointwiseFilterCount: 512,
      depthMultiplier: depthMultiplier,
      strides: (1, 1))
    dConvBlock8 = DepthwiseConvBlock(
      filterCount: 512,
      pointwiseFilterCount: 512,
      depthMultiplier: depthMultiplier,
      strides: (1, 1))
    dConvBlock9 = DepthwiseConvBlock(
      filterCount: 512,
      pointwiseFilterCount: 512,
      depthMultiplier: depthMultiplier,
      strides: (1, 1))
    dConvBlock10 = DepthwiseConvBlock(
      filterCount: 512,
      pointwiseFilterCount: 512,
      depthMultiplier: depthMultiplier,
      strides: (1, 1))
    dConvBlock11 = DepthwiseConvBlock(
      filterCount: 512,
      pointwiseFilterCount: 512,
      depthMultiplier: depthMultiplier,
      strides: (1, 1))
    dConvBlock12 = DepthwiseConvBlock(
      filterCount: 512,
      pointwiseFilterCount: 1024,
      depthMultiplier: depthMultiplier,
      strides: (2, 2))
    dConvBlock13 = DepthwiseConvBlock(
      filterCount: 1024,
      pointwiseFilterCount: 1024,
      depthMultiplier: depthMultiplier,
      strides: (1, 1))

    dropoutLayer = Dropout<Float>(probability: dropout)
    convLast = Conv2D<Float>(
      filterShape: (1, 1, 1024, classCount),
      strides: (1, 1),
      padding: .same)
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    let convolved = input.sequenced(through: convBlock1, dConvBlock1,
      dConvBlock2, dConvBlock3, dConvBlock4)
    let convolved2 = convolved.sequenced(through: dConvBlock5, dConvBlock6,
      dConvBlock7, dConvBlock8, dConvBlock9)
    let convolved3 = convolved2.sequenced(through: dConvBlock10, dConvBlock11,
      dConvBlock12, dConvBlock13)

    // Verify shape since any discrepancies will be masked after pooling
    let convolved4 = convolved3.sequenced(through: avgPool, reshape,
      dropoutLayer, convLast)
    let output = convolved4.reshaped(to: [1, classCount])
    return output
  }  
}

