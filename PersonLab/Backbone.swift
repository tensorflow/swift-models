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
import TensorFlow

public struct DepthwiseSeparableConvBlock: Layer {
  var dConv: DepthwiseConv2D<Float>
  var conv: Conv2D<Float>
  var depthWiseFilter: Tensor<Float>
  var depthWiseBias: Tensor<Float>
  var pointWiseFilter: Tensor<Float>
  var pointWiseBias: Tensor<Float>

  @noDerivative let strides: (Int, Int)

  public init(
    depthWiseFilter: Tensor<Float>,
    depthWiseBias: Tensor<Float>,
    pointWiseFilter: Tensor<Float>,
    pointWiseBias: Tensor<Float>,
    strides: (Int, Int)
  ) {
    self.depthWiseFilter = depthWiseFilter
    self.depthWiseBias = depthWiseBias
    self.pointWiseFilter = pointWiseFilter
    self.pointWiseBias = pointWiseBias
    self.strides = strides

    dConv = DepthwiseConv2D<Float>(
      filter: depthWiseFilter,
      bias: depthWiseBias,
      activation: relu6,
      strides: strides,
      padding: .same
    )

    conv = Conv2D<Float>(
      filter: pointWiseFilter,
      bias: pointWiseBias,
      activation: relu6,
      padding: .same
    )
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    return input.sequenced(through: dConv, conv)
  }
}

public struct MobileNetLikeBackbone: Layer {
  @noDerivative let ckpt: CheckpointReader

  public var convBlock0: Conv2D<Float>
  public var dConvBlock1: DepthwiseSeparableConvBlock
  public var dConvBlock2: DepthwiseSeparableConvBlock
  public var dConvBlock3: DepthwiseSeparableConvBlock
  public var dConvBlock4: DepthwiseSeparableConvBlock
  public var dConvBlock5: DepthwiseSeparableConvBlock
  public var dConvBlock6: DepthwiseSeparableConvBlock
  public var dConvBlock7: DepthwiseSeparableConvBlock
  public var dConvBlock8: DepthwiseSeparableConvBlock
  public var dConvBlock9: DepthwiseSeparableConvBlock
  public var dConvBlock10: DepthwiseSeparableConvBlock
  public var dConvBlock11: DepthwiseSeparableConvBlock
  public var dConvBlock12: DepthwiseSeparableConvBlock
  public var dConvBlock13: DepthwiseSeparableConvBlock

  public init(checkpoint: CheckpointReader) {
    self.ckpt = checkpoint

    self.convBlock0 = Conv2D<Float>(
      filter: ckpt.load(from: "Conv2d_0/weights"),
      bias: ckpt.load(from: "Conv2d_0/biases"),
      activation: relu6,
      strides: (2, 2),
      padding: .same
    )
    self.dConvBlock1 = DepthwiseSeparableConvBlock(
      depthWiseFilter: ckpt.load(from: "Conv2d_1_depthwise/depthwise_weights"),
      depthWiseBias: ckpt.load(from: "Conv2d_1_depthwise/biases"),
      pointWiseFilter: ckpt.load(from: "Conv2d_1_pointwise/weights"),
      pointWiseBias: ckpt.load(from: "Conv2d_1_pointwise/biases"),
      strides: (1, 1)
    )
    self.dConvBlock2 = DepthwiseSeparableConvBlock(
      depthWiseFilter: ckpt.load(from: "Conv2d_2_depthwise/depthwise_weights"),
      depthWiseBias: ckpt.load(from: "Conv2d_2_depthwise/biases"),
      pointWiseFilter: ckpt.load(from: "Conv2d_2_pointwise/weights"),
      pointWiseBias: ckpt.load(from: "Conv2d_2_pointwise/biases"),
      strides: (2, 2)
    )
    self.dConvBlock3 = DepthwiseSeparableConvBlock(
      depthWiseFilter: ckpt.load(from: "Conv2d_3_depthwise/depthwise_weights"),
      depthWiseBias: ckpt.load(from: "Conv2d_3_depthwise/biases"),
      pointWiseFilter: ckpt.load(from: "Conv2d_3_pointwise/weights"),
      pointWiseBias: ckpt.load(from: "Conv2d_3_pointwise/biases"),
      strides: (1, 1)
    )
    self.dConvBlock4 = DepthwiseSeparableConvBlock(
      depthWiseFilter: ckpt.load(from: "Conv2d_4_depthwise/depthwise_weights"),
      depthWiseBias: ckpt.load(from: "Conv2d_4_depthwise/biases"),
      pointWiseFilter: ckpt.load(from: "Conv2d_4_pointwise/weights"),
      pointWiseBias: ckpt.load(from: "Conv2d_4_pointwise/biases"),
      strides: (2, 2)
    )
    self.dConvBlock5 = DepthwiseSeparableConvBlock(
      depthWiseFilter: ckpt.load(from: "Conv2d_5_depthwise/depthwise_weights"),
      depthWiseBias: ckpt.load(from: "Conv2d_5_depthwise/biases"),
      pointWiseFilter: ckpt.load(from: "Conv2d_5_pointwise/weights"),
      pointWiseBias: ckpt.load(from: "Conv2d_5_pointwise/biases"),
      strides: (1, 1)
    )
    self.dConvBlock6 = DepthwiseSeparableConvBlock(
      depthWiseFilter: ckpt.load(from: "Conv2d_6_depthwise/depthwise_weights"),
      depthWiseBias: ckpt.load(from: "Conv2d_6_depthwise/biases"),
      pointWiseFilter: ckpt.load(from: "Conv2d_6_pointwise/weights"),
      pointWiseBias: ckpt.load(from: "Conv2d_6_pointwise/biases"),
      strides: (2, 2)
    )
    self.dConvBlock7 = DepthwiseSeparableConvBlock(
      depthWiseFilter: ckpt.load(from: "Conv2d_7_depthwise/depthwise_weights"),
      depthWiseBias: ckpt.load(from: "Conv2d_7_depthwise/biases"),
      pointWiseFilter: ckpt.load(from: "Conv2d_7_pointwise/weights"),
      pointWiseBias: ckpt.load(from: "Conv2d_7_pointwise/biases"),
      strides: (1, 1)
    )
    self.dConvBlock8 = DepthwiseSeparableConvBlock(
      depthWiseFilter: ckpt.load(from: "Conv2d_8_depthwise/depthwise_weights"),
      depthWiseBias: ckpt.load(from: "Conv2d_8_depthwise/biases"),
      pointWiseFilter: ckpt.load(from: "Conv2d_8_pointwise/weights"),
      pointWiseBias: ckpt.load(from: "Conv2d_8_pointwise/biases"),
      strides: (1, 1)
    )
    self.dConvBlock9 = DepthwiseSeparableConvBlock(
      depthWiseFilter: ckpt.load(from: "Conv2d_9_depthwise/depthwise_weights"),
      depthWiseBias: ckpt.load(from: "Conv2d_9_depthwise/biases"),
      pointWiseFilter: ckpt.load(from: "Conv2d_9_pointwise/weights"),
      pointWiseBias: ckpt.load(from: "Conv2d_9_pointwise/biases"),
      strides: (1, 1)
    )
    self.dConvBlock10 = DepthwiseSeparableConvBlock(
      depthWiseFilter: ckpt.load(from: "Conv2d_10_depthwise/depthwise_weights"),
      depthWiseBias: ckpt.load(from: "Conv2d_10_depthwise/biases"),
      pointWiseFilter: ckpt.load(from: "Conv2d_10_pointwise/weights"),
      pointWiseBias: ckpt.load(from: "Conv2d_10_pointwise/biases"),
      strides: (1, 1)
    )
    self.dConvBlock11 = DepthwiseSeparableConvBlock(
      depthWiseFilter: ckpt.load(from: "Conv2d_11_depthwise/depthwise_weights"),
      depthWiseBias: ckpt.load(from: "Conv2d_11_depthwise/biases"),
      pointWiseFilter: ckpt.load(from: "Conv2d_11_pointwise/weights"),
      pointWiseBias: ckpt.load(from: "Conv2d_11_pointwise/biases"),
      strides: (1, 1)
    )
    self.dConvBlock12 = DepthwiseSeparableConvBlock(
      depthWiseFilter: ckpt.load(from: "Conv2d_12_depthwise/depthwise_weights"),
      depthWiseBias: ckpt.load(from: "Conv2d_12_depthwise/biases"),
      pointWiseFilter: ckpt.load(from: "Conv2d_12_pointwise/weights"),
      pointWiseBias: ckpt.load(from: "Conv2d_12_pointwise/biases"),
      strides: (1, 1)
    )
    self.dConvBlock13 = DepthwiseSeparableConvBlock(
      depthWiseFilter: ckpt.load(from: "Conv2d_13_depthwise/depthwise_weights"),
      depthWiseBias: ckpt.load(from: "Conv2d_13_depthwise/biases"),
      pointWiseFilter: ckpt.load(from: "Conv2d_13_pointwise/weights"),
      pointWiseBias: ckpt.load(from: "Conv2d_13_pointwise/biases"),
      strides: (1, 1)
    )
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    var x = convBlock0(input)
    x = dConvBlock1(x)
    x = dConvBlock2(x)
    x = dConvBlock3(x)
    x = dConvBlock4(x)
    x = dConvBlock5(x)
    x = dConvBlock6(x)
    x = dConvBlock7(x)
    x = dConvBlock8(x)
    x = dConvBlock9(x)
    x = dConvBlock10(x)
    x = dConvBlock11(x)
    x = dConvBlock12(x)
    x = dConvBlock13(x)
    return x
  }

}
