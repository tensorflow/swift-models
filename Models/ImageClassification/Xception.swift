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

public struct ConvBlock: Layer {
  @noDerivative public var depthActivation: Bool
  public var conv: Conv2D<Float>
  public var batchNorm: BatchNorm<Float>

  public init(
    filterShape: (Int, Int, Int, Int),
    strides: (Int, Int) = (1,1),
    padding: Padding = .valid,
    dilations: (Int, Int) = (1,1),
    useBias: Bool = true,
    depthActivation: Bool = true
  ){
    self.depthActivation = depthActivation
    conv = Conv2D<Float>(
      filterShape: filterShape, strides: strides, 
      padding: padding, dilations: dilations, useBias: useBias)
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
  @noDerivative public var depthActivation: Bool
  public var sepConv: SeparableConv2D<Float>
  public var batchNorm: BatchNorm<Float>

  public init(
    filterShape: (Int, Int, Int, Int),
    widthMultiplier: Float = 1.0,
    depthMultiplier: Int = 1,
    strides: (Int, Int) = (1,1), 
    depthActivation: Bool = true
    ) {
      precondition(widthMultiplier > 0, "Width multiplier must be positive")
      precondition(depthMultiplier > 0, "Depth multiplier must be positive")

      let scaledFilterCount = Int(Float(filterShape.2) * widthMultiplier)
      let scaledPointwiseFilterCount = Int(Float(filterShape.3) * widthMultiplier)

      self.depthActivation = depthActivation

      sepConv = SeparableConv2D<Float>(
        depthwiseFilterShape: (filterShape.0, filterShape.1, scaledFilterCount, depthMultiplier),
        pointwiseFilterShape: (1, 1, scaledFilterCount * depthMultiplier, scaledPointwiseFilterCount),
        strides: strides,
        padding: .same,
        useBias: false)
      batchNorm = BatchNorm<Float>(featureCount: scaledPointwiseFilterCount)
      }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    if self.depthActivation {
      return relu(input.sequenced(through: sepConv, batchNorm))
    } else {
      return input.sequenced(through: sepConv, batchNorm)
    }
  }
}


public struct Xception: Layer {
  @noDerivative let classCount: Int
  @noDerivative let includeTop: Bool
  @noDerivative let pooling: String

  public var maxPool = MaxPool2D<Float>(poolSize: (3, 3), strides: (2, 2), padding: .same)
  public var convBlock1: ConvBlock
  public var convBlock2: ConvBlock
  public var residualBlock1: ConvBlock
  public var residualBlock2: ConvBlock
  public var residualBlock3: ConvBlock
  public var residualBlock4: ConvBlock
  public var sepConvBlock1a: SeparableConvBlock
  public var sepConvBlock2a: SeparableConvBlock
  public var sepConvBlock3a: SeparableConvBlock
  public var sepConvBlock4a: SeparableConvBlock
  public var sepConvBlock5a: SeparableConvBlock
  public var sepConvBlock6a: SeparableConvBlock
  public var sepConvBlock1b: SeparableConvBlock
  public var sepConvBlock2b: SeparableConvBlock
  public var sepConvBlock3b: SeparableConvBlock
  public var sepConvBlock1c: SeparableConvBlock
  public var sepConvBlock2c: SeparableConvBlock
  public var sepConvBlock3c: SeparableConvBlock
  public var sepConvBlock4c: SeparableConvBlock
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

    convBlock1 = ConvBlock(filterShape: (3, 3, 3, 32), strides: (2,2))
    convBlock2 = ConvBlock(filterShape: (3, 3, 32, 64))
    residualBlock1 = ConvBlock(filterShape: (1, 1, 64, 128), strides: (2,2), padding: .same, depthActivation: false)

    sepConvBlock1a = SeparableConvBlock(
      filterShape: (3, 3, 64, 128),
      widthMultiplier: widthMultiplier,
      depthMultiplier: depthMultiplier,
      depthActivation: true
    )
    sepConvBlock2a = SeparableConvBlock(
      filterShape: (3, 3, 128, 128),
      widthMultiplier: widthMultiplier,
      depthMultiplier: depthMultiplier,
      depthActivation: false
    )
    
    residualBlock2 = ConvBlock(filterShape: (1, 1, 128, 256), strides: (2,2), padding: .same, depthActivation: false)

    sepConvBlock3a = SeparableConvBlock(
      filterShape: (3, 3, 128, 256),
      widthMultiplier: widthMultiplier,
      depthMultiplier: depthMultiplier,
      depthActivation: true
    )
    sepConvBlock4a = SeparableConvBlock(
      filterShape: (3, 3, 256, 256),
      widthMultiplier: widthMultiplier,
      depthMultiplier: depthMultiplier,
      depthActivation: false
    )

    residualBlock3 = ConvBlock(filterShape: (1, 1, 256, 728), strides: (2,2), padding: .same, depthActivation: false)

    sepConvBlock5a = SeparableConvBlock(
      filterShape: (3, 3, 256, 728),
      widthMultiplier: widthMultiplier,
      depthMultiplier: depthMultiplier,
      depthActivation: true
    )
    sepConvBlock6a = SeparableConvBlock(
      filterShape: (3, 3, 728, 728),
      widthMultiplier: widthMultiplier,
      depthMultiplier: depthMultiplier,
      depthActivation: false
    )
    
    sepConvBlock1b = SeparableConvBlock(
      filterShape: (3, 3, 728, 728),
      widthMultiplier: widthMultiplier,
      depthMultiplier: depthMultiplier,
      depthActivation: true
    )
    sepConvBlock2b = SeparableConvBlock(
      filterShape: (3, 3, 728, 728),
      widthMultiplier: widthMultiplier,
      depthMultiplier: depthMultiplier,
      depthActivation: true
    )
    sepConvBlock3b = SeparableConvBlock(
      filterShape: (3, 3, 728, 728),
      widthMultiplier: widthMultiplier,
      depthMultiplier: depthMultiplier,
      depthActivation: false
    )

    residualBlock4 = ConvBlock(filterShape: (1, 1, 728, 1024), strides: (2,2), padding: .same, depthActivation: false)
    
    sepConvBlock1c = SeparableConvBlock(
      filterShape: (3, 3, 728, 728),
      widthMultiplier: widthMultiplier,
      depthMultiplier: depthMultiplier,
      depthActivation: true
    )
    sepConvBlock2c = SeparableConvBlock(
      filterShape: (3, 3, 728, 1024),
      widthMultiplier: widthMultiplier,
      depthMultiplier: depthMultiplier,
      depthActivation: false
    )

    sepConvBlock3c = SeparableConvBlock(
      filterShape: (3, 3, 1024, 1536),
      widthMultiplier: widthMultiplier,
      depthMultiplier: depthMultiplier,
      depthActivation: true
    )
    sepConvBlock4c = SeparableConvBlock(
      filterShape: (3, 3, 1536, 2048),
      widthMultiplier: widthMultiplier,
      depthMultiplier: depthMultiplier,
      depthActivation: true
    )
    
    denseLast = Dense<Float>(inputSize: 2048, outputSize: classCount, activation: softmax)
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    var convolved = input.sequenced(through: convBlock1, convBlock2)
    var residual: Tensor<Float>
    
    residual = residualBlock1(convolved)
    convolved = convolved.sequenced(through: sepConvBlock1a, sepConvBlock2a, maxPool)
    convolved = convolved + residual
    
    residual = residualBlock2(convolved)
    convolved = relu(convolved)
    convolved = convolved.sequenced(through: sepConvBlock3a, sepConvBlock4a, maxPool)
    convolved = convolved + residual
    
    residual = residualBlock3(convolved)
    convolved = relu(convolved)
    convolved = convolved.sequenced(through: sepConvBlock5a, sepConvBlock6a, maxPool)
    convolved = convolved + residual

    for _ in 0..<8 {
      residual = convolved
      convolved = relu(convolved)
      convolved = convolved.sequenced(through: sepConvBlock1b, sepConvBlock2b, sepConvBlock3b)
      convolved = convolved + residual
    }

    residual = residualBlock4(convolved)
    convolved = relu(convolved)
    convolved = convolved.sequenced(through: sepConvBlock1c, sepConvBlock2c, maxPool)
    convolved = convolved + residual
    convolved = convolved.sequenced(through: sepConvBlock3c, sepConvBlock4c)

    if includeTop {
      convolved = globalAvgPool(convolved)
      convolved = denseLast(convolved)
    } else {
      if pooling == "avg" {
        convolved = globalAvgPool(convolved)
      } else if pooling == "max" {
        convolved = globalMaxPool(convolved)
      }
    }
    return convolved
  }
}
