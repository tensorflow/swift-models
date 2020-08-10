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
// "Wide Residual Networks"
// Sergey Zagoruyko, Nikos Komodakis
// https://arxiv.org/abs/1605.07146
// https://github.com/szagoruyko/wide-residual-networks

public struct BatchNormConv2DBlock: Layer {
    public var norm1: BatchNorm<Float>
    public var conv1: Conv2D<Float>
    public var norm2: BatchNorm<Float>
    public var conv2: Conv2D<Float>
    public var shortcut: Conv2D<Float>
    @noDerivative let isExpansion: Bool
    @noDerivative let dropout: Dropout<Float> = Dropout(probability: 0.3)

    public init(
        featureCounts: (Int, Int),
        kernelSize: Int = 3,
        strides: (Int, Int) = (1, 1),
        padding: Padding = .same
    ) {
        self.norm1 = BatchNorm(featureCount: featureCounts.0)
        self.conv1 = Conv2D(
            filterShape: (kernelSize, kernelSize, featureCounts.0, featureCounts.1), 
            strides: strides, 
            padding: padding)
        self.norm2 = BatchNorm(featureCount: featureCounts.1)
        self.conv2 = Conv2D(filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.1), 
                            strides: (1, 1), 
                            padding: padding)
        self.shortcut = Conv2D(filterShape: (1, 1, featureCounts.0, featureCounts.1), 
                               strides: strides, 
                               padding: padding)
        self.isExpansion = featureCounts.1 != featureCounts.0 || strides != (1, 1) 
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let preact1 = relu(norm1(input))
        var residual = conv1(preact1)
        let preact2: Tensor<Float>
        let shortcutResult: Tensor<Float>
        if isExpansion {
            shortcutResult = shortcut(preact1)
            preact2 = relu(norm2(residual))
        } else { 
            shortcutResult = input
            preact2 = dropout(relu(norm2(residual)))
        }
        residual = conv2(preact2)
        return residual + shortcutResult
    }
}

public struct WideResNetBasicBlock: Layer {
    public var blocks: [BatchNormConv2DBlock]

    public init(
        featureCounts: (Int, Int),
        kernelSize: Int = 3,
        depthFactor: Int = 2,
        initialStride: (Int, Int) = (2, 2)
    ) {
        self.blocks = [BatchNormConv2DBlock(featureCounts: featureCounts, strides: initialStride)]    
        for _ in 1..<depthFactor {
            self.blocks += [BatchNormConv2DBlock(featureCounts: (featureCounts.1, featureCounts.1))]
        }  
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return blocks.differentiableReduce(input) { $1($0) }
    }
}

public struct WideResNet: Layer {
    @_Freezable public var l1: Conv2D<Float>

    @_Freezable public var l2: WideResNetBasicBlock
    @_Freezable public var l3: WideResNetBasicBlock
    @_Freezable public var l4: WideResNetBasicBlock

    public var norm: BatchNorm<Float>
    public var avgPool: AvgPool2D<Float>
    public var flatten = Flatten<Float>()
    @_Freezable public var classifier: Dense<Float>

    public init(depthFactor: Int = 2, widenFactor: Int = 8) {
        self.l1 = Conv2D(filterShape: (3, 3, 3, 16), strides: (1, 1), padding: .same)

        self.l2 = WideResNetBasicBlock(
            featureCounts: (16, 16 * widenFactor), depthFactor: depthFactor, initialStride: (1, 1))
        self.l3 = WideResNetBasicBlock(featureCounts: (16 * widenFactor, 32 * widenFactor), 
                                       depthFactor: depthFactor)
        self.l4 = WideResNetBasicBlock(featureCounts: (32 * widenFactor, 64 * widenFactor), 
                                       depthFactor: depthFactor)

        self.norm = BatchNorm(featureCount: 64 * widenFactor)
        self.avgPool = AvgPool2D(poolSize: (8, 8), strides: (8, 8))
        self.classifier = Dense(inputSize: 64 * widenFactor, outputSize: 10)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let inputLayer = input.sequenced(through: l1, l2, l3, l4)
        let finalNorm = relu(norm(inputLayer))
        return finalNorm.sequenced(through: avgPool, flatten, classifier)
    }
}

extension WideResNet {
    public enum Kind {
        case wideResNet16
        case wideResNet16k8
        case wideResNet16k10
        case wideResNet22
        case wideResNet22k8
        case wideResNet22k10
        case wideResNet28
        case wideResNet28k10
        case wideResNet28k12
        case wideResNet40k1
        case wideResNet40k2
        case wideResNet40k4
        case wideResNet40k8
    }

    public init(kind: Kind) {
        switch kind {
        case .wideResNet16, .wideResNet16k8:
            self.init(depthFactor: 2, widenFactor: 8)
        case .wideResNet16k10:
            self.init(depthFactor: 2, widenFactor: 10)
        case .wideResNet22, .wideResNet22k8:
            self.init(depthFactor: 3, widenFactor: 8)
        case .wideResNet22k10:
            self.init(depthFactor: 3, widenFactor: 10)
        case .wideResNet28, .wideResNet28k10:
            self.init(depthFactor: 4, widenFactor: 10)
        case .wideResNet28k12:
            self.init(depthFactor: 4, widenFactor: 12)
        case .wideResNet40k1:
            self.init(depthFactor: 6, widenFactor: 1)
        case .wideResNet40k2:
            self.init(depthFactor: 6, widenFactor: 2)
        case .wideResNet40k4:
            self.init(depthFactor: 6, widenFactor: 4)
        case .wideResNet40k8:
            self.init(depthFactor: 6, widenFactor: 8)
        }
    }
}
