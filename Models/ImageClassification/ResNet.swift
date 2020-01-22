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
import Foundation

// Original Paper:
// "Deep Residual Learning for Image Recognition"
// Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
// https://arxiv.org/abs/1512.03385
// This uses shortcut layers to connect residual blocks (aka Option (B)).

public protocol Projection: Module {
    @differentiable func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float>    
}

public struct Passthrough: ParameterlessLayer, Projection {
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input
    }
}

public struct ConvBN: Layer, Projection {
    public var conv: Conv2D<Float>
    public var norm: BatchNorm<Float>

    public init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid
    ) {
        self.conv = Conv2D(filterShape: filterShape, strides: strides, padding: padding)
        // TODO: zero scale initialization for last BN
        self.norm = BatchNorm(featureCount: filterShape.3, momentum: 0.9, epsilon: 1e-5)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: conv, norm)
    }
}

public struct ResidualBlock: Layer {
    public var projection: ConvBN
    @noDerivative public let needsProjection: Bool
    public var earlyConvs: [ConvBN] = []
    public var lastConv: ConvBN

    public init(inputFilters: Int, filters: Int, strides: (Int, Int), useLaterStride: Bool, isBasic: Bool) {
        self.needsProjection = (inputFilters != (filters * 4)) || (strides.0 != 1)
        // TODO: Replace the following, so as to not waste memory for non-projection cases.
        projection = ConvBN(filterShape: (1, 1, inputFilters, filters * 4), strides: strides)

        if isBasic {
            earlyConvs = [(ConvBN(filterShape: (3, 3, inputFilters, filters), strides: strides))]
            lastConv = ConvBN(filterShape: (3, 3, filters, filters * 4)) // TODO: 0-scale for this BN
        } else {
            if useLaterStride { // ResNet V1.5
                earlyConvs.append(ConvBN(filterShape: (1, 1, inputFilters, filters)))
                earlyConvs.append(ConvBN(filterShape: (3, 3, filters, filters), strides: strides))
            } else { // ResNet V1
                print("1 input: \(inputFilters), output: \(filters)")
                earlyConvs.append(ConvBN(filterShape: (1, 1, inputFilters, filters), strides: strides))
                print("2 input: \(filters), output: \(filters)")
                earlyConvs.append(ConvBN(filterShape: (3, 3, filters, filters)))
            }
            print("3 input: \(filters), output: \(filters * 4)")
            lastConv = ConvBN(filterShape: (1, 1, filters, filters * 4)) // TODO: 0-scale for this BN
        }
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        print("block input: \(input.shape)")

        let residual: Tensor<Float>
        if needsProjection {
            residual = projection(input)
        } else {
            residual = input
        }
        
        let earlyConvsReduced = earlyConvs.differentiableReduce(input) { last, layer in
            relu(layer(last))
        }
        let lastConvResult = lastConv(earlyConvsReduced)
        print("lastConvResult: \(lastConvResult.shape), residual: \(residual.shape)")

        return relu(lastConvResult + residual)
    }
}

public struct ResNet: Layer {
    public var initialLayer: ConvBN
    public var maxPool: MaxPool2D<Float>

    public var residualBlocks: [ResidualBlock] = []

    public var avgPool = GlobalAvgPool2D<Float>()
    public var flatten = Flatten<Float>()
    public var classifier: Dense<Float>

    // useLaterStride enables the ResNet V1.5 variant.
    public init(classCount: Int, depth: Depth, imageSize: ImageSize, inputFilters: Int = 64, useLaterStride: Bool = false) {
        switch imageSize {
        case .imagenet:
            initialLayer = ConvBN(filterShape: (7, 7, 3, inputFilters), strides: (2, 2), padding: .same)
            maxPool = MaxPool2D(poolSize: (3, 3), strides: (2, 2))
        case .cifar:
            // initialLayer = ConvBN(filterShape: (3, 3, 3, inputFilters), padding: .same)
            initialLayer = ConvBN(filterShape: (3, 3, 3, inputFilters), padding: .valid)
            maxPool = MaxPool2D(poolSize: (1, 1), strides: (1, 1))  // no-op
        }

        var lastInputFilterCount = inputFilters
        for (blockSizeIndex, blockSize) in depth.layerBlockSizes.enumerated() {
            for blockIndex in 0..<blockSize {
                let strides = ((blockSizeIndex > 0) && (blockIndex == 0) ) ? (2, 2) : (1, 1)
                let filters = inputFilters * Int(pow(2.0, Double(blockSizeIndex)))
                let residualBlock = ResidualBlock(inputFilters: lastInputFilterCount, filters: filters, strides: strides, useLaterStride: useLaterStride, isBasic: depth.usesBasicBlocks)
                lastInputFilterCount = filters * 4
                residualBlocks.append(residualBlock)
            }
        }

        classifier = Dense(inputSize: 2048, outputSize: classCount)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let inputLayer = maxPool(relu(initialLayer(input)))
        print("inputLayer: \(inputLayer.shape)")
        let blocksReduced = residualBlocks.differentiableReduce(inputLayer) { last, layer in
            layer(last)
        }
        return blocksReduced.sequenced(through: avgPool, flatten, classifier)
    }
}


extension ResNet {
    public enum ImageSize {
        case cifar
        case imagenet
    }

    public enum Depth {
        case resNet18
        case resNet34
        case resNet50
        case resNet101
        case resNet152

        var usesBasicBlocks: Bool {
            switch self {
            case .resNet18, .resNet34: return true
            default: return false
            }
        }

        var layerBlockSizes: [Int] {
            switch self {
            case .resNet18: return [2, 2, 2, 2]
            case .resNet34: return [3, 4, 6, 3]
            case .resNet50: return [3, 4, 6, 3]
            case .resNet101: return [3, 4, 23, 3]
            case .resNet152: return [3, 8, 36, 3]
            }
        }
    }
}
