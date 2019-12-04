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
  public var zeroPad = ZeroPadding2D<Float>(
      padding: ((0, 1), (0, 1)))
  public var conv: Conv2D<Float>
  public var batchNorm: BatchNorm<Float>

  public init(featureCount: Int, filterShape: (Int, Int, Int, Int), strides: (Int, Int)) {
    conv = Conv2D<Float>(
      filterShape: filterShape,
      strides: strides,
      padding: .valid)
    batchNorm = BatchNorm<Float>(featureCount: featureCount)
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    // let convolved = input.sequenced(through: zeroPad, conv, batchNorm)
    // return relu6(convolved)
    let c1 = zeroPad(input)
//    print("zeroPad: \(c1.shape)")
    let c2 = conv(c1)
//    print("conv: \(c2.shape)")
    let c3 = batchNorm(c2)
//    print("batchNorm: \(c3.shape)")
    let c4 = relu(c3)
//    print("relu: \(c4.shape)")
    return c4
  }

}

public struct DepthwiseConvBlock: Layer {
  public var zeroPad = ZeroPadding2D<Float>(
      padding: ((0, 1), (0, 1)))
  public var dConv: DepthwiseConv2D<Float>
  public var batchNorm1: BatchNorm<Float>
  public var conv: Conv2D<Float>
  public var batchNorm2: BatchNorm<Float>

  public init(inputChannel: Int, outputChannel: Int, strides: (Int, Int)) {
    dConv = DepthwiseConv2D<Float>(
      filterShape: (3, 3, inputChannel, 1),
      strides: strides,
      padding: .same)
    batchNorm1 = BatchNorm<Float>(featureCount: inputChannel)
    conv = Conv2D<Float>(
      filterShape: (1, 1, inputChannel, outputChannel),
      strides: (1, 1),
      padding: .same)
    batchNorm2 = BatchNorm<Float>(featureCount: outputChannel)
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    // let convolved1 = input.sequenced(through: zeroPad, dConv, batchNorm1)
    // let convolved2 = relu6(convolved1).sequenced(through: conv, batchNorm2)
    // return relu6(convolved2)
    let c1 = dConv(input)
 //   print("dConv: \(c1.shape)")
    let c2 = batchNorm1(c1)
 //   print("batchNorm1: \(c2.shape)")
    let c2_2 = relu6(c2)
 //   print("relu6: \(c2_2.shape)")
    let c3 = conv(c2_2)
 //   print("conv: \(c3.shape)")
    let c4 = batchNorm2(c3)
 //   print("batchNorm2: \(c4.shape)")
    let c5 = relu6(c4)
 //   print("relu6: \(c5.shape)")
    return c5
  }

}

public struct MobileNetV1: Layer {
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
  public var dropout = Dropout<Float>(probability: 0.001)
  public var convLast: Conv2D<Float>
  public var dense: Dense<Float>

  public init(classCount: Int) {
  //  convBlock1 = ConvBlock(featureCount: classCount, filterShape: (1, 1, 224, 32), strides: (2, 2))
    convBlock1 = ConvBlock(featureCount: 32, filterShape: (3, 3, 3, 32), strides: (2, 2))
    dConvBlock1 = DepthwiseConvBlock(inputChannel: 32, outputChannel: 64, strides: (1, 1))
    dConvBlock2 = DepthwiseConvBlock(inputChannel: 64, outputChannel: 128, strides: (2, 2))
    dConvBlock3 = DepthwiseConvBlock(inputChannel: 128, outputChannel: 128, strides: (1, 1))
    dConvBlock4 = DepthwiseConvBlock(inputChannel: 128, outputChannel: 256, strides: (2, 2))
    dConvBlock5 = DepthwiseConvBlock(inputChannel: 256, outputChannel: 256, strides: (1, 1))
    dConvBlock6 = DepthwiseConvBlock(inputChannel: 256, outputChannel: 512, strides: (2, 2))
    dConvBlock7 = DepthwiseConvBlock(inputChannel: 512, outputChannel: 512, strides: (1, 1))
    dConvBlock8 = DepthwiseConvBlock(inputChannel: 512, outputChannel: 512, strides: (1, 1))
    dConvBlock9 = DepthwiseConvBlock(inputChannel: 512, outputChannel: 512, strides: (1, 1))
    dConvBlock10 = DepthwiseConvBlock(inputChannel: 512, outputChannel: 512, strides: (1, 1))
    dConvBlock11 = DepthwiseConvBlock(inputChannel: 512, outputChannel: 512, strides: (1, 1))
    dConvBlock12 = DepthwiseConvBlock(inputChannel: 512, outputChannel: 1024, strides: (2, 2))
    dConvBlock13 = DepthwiseConvBlock(inputChannel: 1024, outputChannel: 1024, strides: (1, 1))

    // TODO(michellecasbon): Add remaining layers
    convLast = Conv2D<Float>(
      filterShape: (1, 1, 1024, classCount),
      strides: (1, 1),
      padding: .same,
      activation: softmax)
    dense = Dense<Float>(inputSize: 32, outputSize: 32)
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    // let convolved = input.sequenced(through: convBlock1, dConvBlock1, dConvBlock2, convLast)
    //return softmax(convolved)
    print("input: \(input.shape)")
    let convolved = convBlock1(input)
    print("convBlock1: \(convolved.shape)")
    let c2 = dConvBlock1(convolved)
    print("dConvBlock1: \(c2.shape)")
    let c3 = dConvBlock2(c2)
    print("dConvBlock2: \(c3.shape)")
    let c4 = dConvBlock3(c3)
    print("dConvBlock3: \(c4.shape)")
    let c5 = dConvBlock4(c4)
    print("dConvBlock4: \(c5.shape)")
    let c6 = dConvBlock5(c5)
    print("dConvBlock5: \(c6.shape)")

    let c7 = c6.sequenced(through: dConvBlock6, dConvBlock7, dConvBlock8, dConvBlock9)
    print("dConvBlock9: \(c7.shape)")
    let c8 = c7.sequenced(through: dConvBlock10, dConvBlock11, dConvBlock12, dConvBlock13)
    print("dConvBlock13: \(c8.shape)")
 
    let c9 = avgPool(c8)
    print("avgPool: \(c9.shape)")
    let c9_2 = reshape(c9)
    print("reshape: \(c9_2.shape)")
    let c10 = dropout(c9_2)
    print("dropout: \(c10.shape)")
    let c11 = convLast(c10)
    print("convLast: \(c11.shape)")
    let c12 = softmax(c11)
    print("softmax: \(c12.shape)")
    let c13 = c12.reshaped(to: [1, 1000])
    print("reshaped: \(c13.shape)")
  
    return c13
  }  
}

