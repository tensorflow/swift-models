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
// "Xception: Deep Learning with Depthwise Separable Convolutions"
// François Chollet
// https://arxiv.org/abs/1610.02357

public struct ConvBlockModule: Layer {
  @noDerivative public var depthActivation: Bool
  public var conv: Conv2D<Float>
  public var batchNorm: BatchNorm<Float>

  public init(
    filterShape: (Int, Int, Int, Int),
    strides: (Int, Int) = (1,1),
    padding: Padding = .valid,
    dilations: (Int, Int) = (1,1),
    depthActivation: Bool = true
  ){
    self.depthActivation = depthActivation
    conv = Conv2D<Float>(
      filterShape: filterShape, strides: strides,
      padding: padding, dilations: dilations, useBias: false)
    batchNorm = BatchNorm<Float>(featureCount: filterShape.3)
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float>{
    let convolved = input.sequenced(through: conv, batchNorm)
    if self.depthActivation {
      return relu(convolved)
    }  else {return convolved}
  }
}


public struct SeparableConvBlock: Layer {
  @noDerivative public var startWithRelu: Bool
  @noDerivative public var depthActivation: Bool
  public var sepConv: SeparableConv2D<Float>
  public var batchNorm: BatchNorm<Float>

  public init(
    filterShape: (Int, Int, Int, Int),
    strides: (Int, Int) = (1,1),
    startWithRelu: Bool = true,
    depthActivation: Bool = false
    ) {
      self.startWithRelu = startWithRelu
      self.depthActivation = depthActivation

      sepConv = SeparableConv2D<Float>(
        depthwiseFilterShape: (filterShape.0, filterShape.1, filterShape.2, 1),
        pointwiseFilterShape: (1, 1, filterShape.2, filterShape.3),
        strides: strides,
        padding: .same,
        useBias: false)
      batchNorm = BatchNorm<Float>(featureCount: filterShape.3)
      }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    var convolve = input
    if self.startWithRelu {
      convolve = relu(input)
    }
    convolve = input.sequenced(through: sepConv, batchNorm)

    if self.depthActivation {
      return relu(convolve)
    }
    else {return convolve}
  }
}


public struct MiddleFlow: Layer {
  public var middleBlock: [SeparableConvBlock] = []

  public init() {
    middleBlock.append(SeparableConvBlock(filterShape: (3, 3, 728, 728)))
    middleBlock.append(SeparableConvBlock(filterShape: (3, 3, 728, 728)))
    middleBlock.append(SeparableConvBlock(filterShape: (3, 3, 728, 728)))
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    return middleBlock.differentiableReduce(input) {$1($0)}
  }
}


public struct Xception: Layer {
  @noDerivative let classCount: Int
  @noDerivative let includeTop: Bool
  @noDerivative let pooling: String

  public var maxPool = MaxPool2D<Float>(poolSize: (3, 3), strides: (2, 2), padding: .same)
  public var convBlock1: ConvBlockModule
  public var convBlock2: ConvBlockModule
  public var residualBlock: [ConvBlockModule] = []
  public var sepConvEntryBlock: [SeparableConvBlock] = []
  public var sepConvMiddleBlock: [MiddleFlow] = []
  public var sepConvExitBlock: [SeparableConvBlock] = []
  
  public var denseLast: Dense<Float>
  public var globalAvgPool = GlobalAvgPool2D<Float>()
  public var globalMaxPool = GlobalMaxPool2D<Float>()

  public init(
    classCount: Int,
    widthMultiplier: Float = 1.0,
    depthMultiplier: Int = 1,
    includeTop: Bool = true,
    pooling: String = "max"
  ) {

    self.classCount = classCount
    self.includeTop = includeTop
    self.pooling = pooling

    // Entry Flow
    convBlock1 = ConvBlockModule(filterShape: (3, 3, 3, 32), strides: (2, 2))
    convBlock2 = ConvBlockModule(filterShape: (3, 3, 32, 64))

    residualBlock.append(ConvBlockModule(
      filterShape: (1, 1, 64, 128),
      strides: (2, 2),
      padding: .same,
      depthActivation: false))
    
    sepConvEntryBlock.append(SeparableConvBlock(
      filterShape: (3, 3, 64, 128),
      startWithRelu: false))

    sepConvEntryBlock.append(SeparableConvBlock(filterShape: (3, 3, 128, 128)))

    residualBlock.append(ConvBlockModule(
      filterShape: (1, 1, 128, 256),
      strides: (2, 2),
      padding: .same,
      depthActivation: false))

    sepConvEntryBlock.append(SeparableConvBlock(filterShape: (3, 3, 128, 256)))
    sepConvEntryBlock.append(SeparableConvBlock(filterShape: (3, 3, 256, 256)))

    residualBlock.append(ConvBlockModule(
      filterShape: (1, 1, 256, 728),
      strides: (2, 2),
      padding: .same,
      depthActivation: false))

    sepConvEntryBlock.append(SeparableConvBlock(filterShape: (3, 3, 256, 728)))
    sepConvEntryBlock.append(SeparableConvBlock(filterShape: (3, 3, 728, 728)))

    // Middle Flow
    sepConvMiddleBlock = Array(repeating: MiddleFlow(), count: 8)

    // Exit Flow
    residualBlock.append(ConvBlockModule(
      filterShape: (1, 1, 728, 1024),
      strides: (2, 2),
      padding: .same,
      depthActivation: false))
    
    sepConvExitBlock.append(SeparableConvBlock(filterShape: (3, 3, 728, 728)))
    sepConvExitBlock.append(SeparableConvBlock(filterShape: (3, 3, 728, 1024)))
    
    sepConvExitBlock.append(SeparableConvBlock(
      filterShape: (3, 3, 1024, 1536),
      startWithRelu: false,
      depthActivation: true))
    
    sepConvExitBlock.append(SeparableConvBlock(
      filterShape: (3, 3, 1536, 2048),
      startWithRelu: false,
      depthActivation: true))
    
    denseLast = Dense<Float>(inputSize: 2048, outputSize: classCount)
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    var entryFlow: Tensor<Float>
    var residual: Tensor<Float>
    entryFlow = input.sequenced(through: convBlock1, convBlock2)

    // Block 1
    residual = residualBlock[0](entryFlow)
    entryFlow = entryFlow.sequenced(through: sepConvEntryBlock[0], sepConvEntryBlock[1], maxPool)
    entryFlow = entryFlow + residual

    // Block 2
    residual = residualBlock[1](entryFlow)
    entryFlow = entryFlow.sequenced(through: sepConvEntryBlock[2], sepConvEntryBlock[3], maxPool)
    entryFlow = entryFlow + residual

    // Block 3
    residual = residualBlock[2](entryFlow)
    entryFlow = entryFlow.sequenced(through: sepConvEntryBlock[4], sepConvEntryBlock[5], maxPool)
    entryFlow = entryFlow + residual
    
    // Middle Flow
    var middleFlow = entryFlow
    for idx in 0..<8 {
      residual = middleFlow
      middleFlow = sepConvMiddleBlock[idx](middleFlow)
      middleFlow = middleFlow + residual
    }

    // Exit Flow
    var exitFlow = middleFlow
    residual = residualBlock[3](exitFlow)
    exitFlow = exitFlow.sequenced(through: sepConvExitBlock[0], sepConvExitBlock[1], maxPool)
    exitFlow = exitFlow + residual

    exitFlow = exitFlow.sequenced(through: sepConvExitBlock[2], sepConvExitBlock[3])

    if self.includeTop {
      exitFlow = globalAvgPool(exitFlow)
      exitFlow = denseLast(exitFlow)
    }
    else {
      if self.pooling == "avg" {
        exitFlow = globalAvgPool(exitFlow)
      } else if pooling == "max" {
        exitFlow = globalMaxPool(exitFlow)
      }
    }
    return exitFlow
  }
}
