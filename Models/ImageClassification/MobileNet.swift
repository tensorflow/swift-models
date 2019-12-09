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
  var strides: (Int, Int)

  public var zeroPad = ZeroPadding2D<Float>(padding: ((0, 1), (0, 1)))
  public var dConv: DepthwiseConv2D<Float>
  public var batchNorm1: BatchNorm<Float>
  public var conv: Conv2D<Float>
  public var batchNorm2: BatchNorm<Float>

  public init(filterCount: Int, pointwiseFilterCount: Int, strides: (Int, Int)) {
    self.strides = strides
    dConv = DepthwiseConv2D<Float>(
      filterShape: (3, 3, filterCount, 1),
      strides: strides,
      padding: .same)
    batchNorm1 = BatchNorm<Float>(featureCount: filterCount)
    conv = Conv2D<Float>(
      filterShape: (1, 1, filterCount, pointwiseFilterCount),
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
  var classCount: Int

  public var convBlock1 = ConvBlock(filterCount: 32, strides: (2, 2))
  public var dConvBlock1 = DepthwiseConvBlock(filterCount: 32, pointwiseFilterCount: 64, strides: (1, 1))
  public var dConvBlock2 = DepthwiseConvBlock(filterCount: 64, pointwiseFilterCount: 128, strides: (2, 2))
  public var dConvBlock3 = DepthwiseConvBlock(filterCount: 128, pointwiseFilterCount: 128, strides: (1, 1))
  public var dConvBlock4 = DepthwiseConvBlock(filterCount: 128, pointwiseFilterCount: 256, strides: (2, 2))
  public var dConvBlock5 = DepthwiseConvBlock(filterCount: 256, pointwiseFilterCount: 256, strides: (1, 1))
  public var dConvBlock6 = DepthwiseConvBlock(filterCount: 256, pointwiseFilterCount: 512, strides: (2, 2))
  public var dConvBlock7 = DepthwiseConvBlock(filterCount: 512, pointwiseFilterCount: 512, strides: (1, 1))
  public var dConvBlock8 = DepthwiseConvBlock(filterCount: 512, pointwiseFilterCount: 512, strides: (1, 1))
  public var dConvBlock9 = DepthwiseConvBlock(filterCount: 512, pointwiseFilterCount: 512, strides: (1, 1))
  public var dConvBlock10 = DepthwiseConvBlock(filterCount: 512, pointwiseFilterCount: 512, strides: (1, 1))
  public var dConvBlock11 = DepthwiseConvBlock(filterCount: 512, pointwiseFilterCount: 512, strides: (1, 1))
  public var dConvBlock12 = DepthwiseConvBlock(filterCount: 512, pointwiseFilterCount: 1024, strides: (2, 2))
  public var dConvBlock13 = DepthwiseConvBlock(filterCount: 1024, pointwiseFilterCount: 1024, strides: (1, 1))
  public var avgPool = GlobalAvgPool2D<Float>()
  public var reshape = Reshape<Float>(shape: [1, 1, 1, 1024])
  public var dropout = Dropout<Float>(probability: 0.001)
  public var convLast: Conv2D<Float>

  public init(classCount: Int) {
    self.classCount = classCount
    convLast = Conv2D<Float>(
      filterShape: (1, 1, 1024, classCount),
      strides: (1, 1),
      padding: .same)
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    let convolved = input.sequenced(through: convBlock1, dConvBlock1, dConvBlock2, dConvBlock3, dConvBlock4)
    let convolved2 = convolved.sequenced(through: dConvBlock5, dConvBlock6, dConvBlock7, dConvBlock8, dConvBlock9)
    let convolved3 = convolved2.sequenced(through: dConvBlock10, dConvBlock11, dConvBlock12, dConvBlock13)
    let convolved4 = convolved3.sequenced(through: avgPool, reshape, dropout, convLast)
    let output = convolved4.reshaped(to: [1, classCount])
    return output
  }  
}

