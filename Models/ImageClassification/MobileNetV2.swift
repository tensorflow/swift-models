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
// "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
// Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
// https://arxiv.org/abs/1801.04381

public struct InitialInvertedBottleneckBlock: Layer {
    public var dConv: DepthwiseConv2D<Float>
    public var batchNormDConv: BatchNorm<Float>
    public var conv2: Conv2D<Float>
    public var batchNormConv: BatchNorm<Float>

    public init(filters: (Int, Int)) {
        dConv = DepthwiseConv2D<Float>(
            filterShape: (3, 3, filters.0, 1),
            strides: (1, 1),
            padding: .same)
        conv2 = Conv2D<Float>(
            filterShape: (1, 1, filters.0, filters.1),
            strides: (1, 1),
            padding: .same)
        batchNormDConv = BatchNorm(featureCount: filters.0)
        batchNormConv = BatchNorm(featureCount: filters.1)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let dw = relu6(batchNormDConv(dConv(input)))
        return relu6(conv2(dw))
    }
}

public struct InvertedResidualBlock: Layer {
    @noDerivative public var addResLayer: Bool
    @noDerivative public var strides: (Int, Int)
    @noDerivative public let zeroPad = ZeroPadding2D<Float>(padding: ((0, 1), (0, 1)))

    public var conv1: Conv2D<Float>
    public var batchNormConv1: BatchNorm<Float>
    public var dConv: DepthwiseConv2D<Float>
    public var batchNormDConv: BatchNorm<Float>
    public var conv2: Conv2D<Float>
    public var batchNormConv2: BatchNorm<Float>

    public init(filters: (Int, Int), depthMultiplier: Int = 6, strides: (Int, Int) = (1, 1)) {
        self.strides = strides
        self.addResLayer = filters.0 == filters.1 && strides == (1, 1)

        let hiddenDimension = filters.0 * depthMultiplier
        conv1 = Conv2D<Float>(
            filterShape: (1, 1, filters.0, hiddenDimension),
            strides: (1, 1),
            padding: .same)
        dConv = DepthwiseConv2D<Float>(
            filterShape: (3, 3, hiddenDimension, 1),
            strides: strides,
            padding: strides == (1, 1) ? .same : .valid)
        conv2 = Conv2D<Float>(
            filterShape: (1, 1, hiddenDimension, filters.1),
            strides: (1, 1),
            padding: .same)
        batchNormConv1 = BatchNorm(featureCount: hiddenDimension)
        batchNormDConv = BatchNorm(featureCount: hiddenDimension)
        batchNormConv2 = BatchNorm(featureCount: filters.1)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let pw = relu6(batchNormConv1(conv1(input)))
        var dw: Tensor<Float>
        if self.strides == (1, 1) {
            dw = batchNormDConv(dConv(pw))
        } else {
            dw = zeroPad(batchNormDConv(dConv(pw)))
        }
        let pwLinear = batchNormConv2(conv2(dw))

        if self.addResLayer {
            return input + pwLinear
        } else {
            return pwLinear
        }
    }
}

public struct InvertedResidualBlockStack: Layer {
    var blocks: [InvertedResidualBlock] = []

    public init(filters: (Int, Int), blockCount: Int, initialStrides: (Int, Int) = (2, 2)) {
        self.blocks = [InvertedResidualBlock(filters: (filters.0, filters.1),
            strides: initialStrides)]
        for _ in 1..<blockCount {
            self.blocks.append(InvertedResidualBlock(filters: (filters.1, filters.1)))
        }
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return blocks.differentiableReduce(input) { $1($0) }
    }
}

public struct MobileNetV2: Layer {
    public var inputConv: Conv2D<Float>
    public var inputConvBatchNorm: BatchNorm<Float>
    public var initialInvertedBottleneck = InitialInvertedBottleneckBlock(filters: (32, 16))

    public var residualBlockStack1 = InvertedResidualBlockStack(filters: (16, 24), blockCount: 2)
    public var residualBlockStack2 = InvertedResidualBlockStack(filters: (24, 32), blockCount: 3)
    public var residualBlockStack3 = InvertedResidualBlockStack(filters: (32, 64), blockCount: 4)
    public var residualBlockStack4 = InvertedResidualBlockStack(filters: (64, 96),
        blockCount: 3, initialStrides: (1, 1))
    public var residualBlockStack5 = InvertedResidualBlockStack(filters: (96, 160), blockCount: 3)

    public var invertedBottleneckBlock16 = InvertedResidualBlock(filters: (160, 320))
    public var finalConv: Conv2D<Float>
    public var avgPool = GlobalAvgPool2D<Float>()
    public var output: Dense<Float>

    public init(classCount: Int = 1000) {
        inputConv = Conv2D<Float>(
            filterShape: (3, 3, 3, 32),
            strides: (2, 2),
            padding: .same)
        inputConvBatchNorm = BatchNorm(featureCount: 32)

        finalConv = Conv2D<Float>(
            filterShape: (1, 1, 320, 1280),
            strides: (1, 1),
            padding: .same)
        output = Dense(inputSize: 1280, outputSize: classCount)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let convolved = input.sequenced(through: inputConv, inputConvBatchNorm,
            initialInvertedBottleneck)
        let backbone = convolved.sequenced(through: residualBlockStack1, residualBlockStack2,
            residualBlockStack3, residualBlockStack4, residualBlockStack5)
        return backbone.sequenced(through: invertedBottleneckBlock16, finalConv, avgPool, output)
    }
}
