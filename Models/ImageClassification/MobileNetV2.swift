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

fileprivate func makeDivisible(filter: Int, widthMultiplier: Float = 1.0, divisor: Float = 8.0)
    -> Int
{
    /// Return a filter multiplied by width, evenly divisible by the divisor
    let filterMult = Float(filter) * widthMultiplier
    let filterAdd = Float(filterMult) + (divisor / 2.0)
    var div = filterAdd / divisor
    div.round(.down)
    div = div * Float(divisor)
    var newFilterCount = max(1, Int(div))
    if newFilterCount < Int(0.9 * Float(filter)) {
        newFilterCount += Int(divisor)
    }
    return Int(newFilterCount)
}

fileprivate func roundFilterPair(filters: (Int, Int), widthMultiplier: Float) -> (Int, Int) {
    return (
        makeDivisible(filter: filters.0, widthMultiplier: widthMultiplier),
        makeDivisible(filter: filters.1, widthMultiplier: widthMultiplier)
    )
}

public struct InitialInvertedBottleneckBlock: Layer {
    public var dConv: DepthwiseConv2D<Float>
    public var batchNormDConv: BatchNorm<Float>
    public var conv2: Conv2D<Float>
    public var batchNormConv: BatchNorm<Float>

    public init(filters: (Int, Int), widthMultiplier: Float) {
        let filterMult = roundFilterPair(filters: filters, widthMultiplier: widthMultiplier)
        dConv = DepthwiseConv2D<Float>(
            filterShape: (3, 3, filterMult.0, 1),
            strides: (1, 1),
            padding: .same)
        conv2 = Conv2D<Float>(
            filterShape: (1, 1, filterMult.0, filterMult.1),
            strides: (1, 1),
            padding: .same)
        batchNormDConv = BatchNorm(featureCount: filterMult.0)
        batchNormConv = BatchNorm(featureCount: filterMult.1)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let depthwise = relu6(batchNormDConv(dConv(input)))
        return batchNormConv(conv2(depthwise))
    }
}

public struct InvertedBottleneckBlock: Layer {
    @noDerivative public var addResLayer: Bool
    @noDerivative public var strides: (Int, Int)
    @noDerivative public let zeroPad = ZeroPadding2D<Float>(padding: ((0, 1), (0, 1)))

    public var conv1: Conv2D<Float>
    public var batchNormConv1: BatchNorm<Float>
    public var dConv: DepthwiseConv2D<Float>
    public var batchNormDConv: BatchNorm<Float>
    public var conv2: Conv2D<Float>
    public var batchNormConv2: BatchNorm<Float>

    public init(
        filters: (Int, Int),
        widthMultiplier: Float,
        depthMultiplier: Int = 6,
        strides: (Int, Int) = (1, 1)
    ) {
        self.strides = strides
        self.addResLayer = filters.0 == filters.1 && strides == (1, 1)

        let filterMult = roundFilterPair(filters: filters, widthMultiplier: widthMultiplier)
        let hiddenDimension = filterMult.0 * depthMultiplier
        conv1 = Conv2D<Float>(
            filterShape: (1, 1, filterMult.0, hiddenDimension),
            strides: (1, 1),
            padding: .same)
        dConv = DepthwiseConv2D<Float>(
            filterShape: (3, 3, hiddenDimension, 1),
            strides: strides,
            padding: strides == (1, 1) ? .same : .valid)
        conv2 = Conv2D<Float>(
            filterShape: (1, 1, hiddenDimension, filterMult.1),
            strides: (1, 1),
            padding: .same)
        batchNormConv1 = BatchNorm(featureCount: hiddenDimension)
        batchNormDConv = BatchNorm(featureCount: hiddenDimension)
        batchNormConv2 = BatchNorm(featureCount: filterMult.1)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let pointwise = relu6(batchNormConv1(conv1(input)))
        var depthwise: Tensor<Float>
        if self.strides == (1, 1) {
            depthwise = relu6(batchNormDConv(dConv(pointwise)))
        } else {
            depthwise = relu6(batchNormDConv(dConv(zeroPad(pointwise))))
        }
        let pointwiseLinear = batchNormConv2(conv2(depthwise))

        if self.addResLayer {
            return input + pointwiseLinear
        } else {
            return pointwiseLinear
        }
    }
}

public struct InvertedBottleneckBlockStack: Layer {
    var blocks: [InvertedBottleneckBlock] = []

    public init(
        filters: (Int, Int),
        widthMultiplier: Float,
        blockCount: Int,
        initialStrides: (Int, Int) = (2, 2)
    ) {
        self.blocks = [
            InvertedBottleneckBlock(
                filters: (filters.0, filters.1), widthMultiplier: widthMultiplier,
                strides: initialStrides)
        ]
        for _ in 1..<blockCount {
            self.blocks.append(
                InvertedBottleneckBlock(
                    filters: (filters.1, filters.1), widthMultiplier: widthMultiplier)
            )
        }
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return blocks.differentiableReduce(input) { $1($0) }
    }
}

public struct MobileNetV2: Layer {
    @noDerivative public let zeroPad = ZeroPadding2D<Float>(padding: ((0, 1), (0, 1)))
    public var inputConv: Conv2D<Float>
    public var inputConvBatchNorm: BatchNorm<Float>
    public var initialInvertedBottleneck: InitialInvertedBottleneckBlock

    public var residualBlockStack1: InvertedBottleneckBlockStack
    public var residualBlockStack2: InvertedBottleneckBlockStack
    public var residualBlockStack3: InvertedBottleneckBlockStack
    public var residualBlockStack4: InvertedBottleneckBlockStack
    public var residualBlockStack5: InvertedBottleneckBlockStack

    public var invertedBottleneckBlock16: InvertedBottleneckBlock

    public var outputConv: Conv2D<Float>
    public var outputConvBatchNorm: BatchNorm<Float>
    public var avgPool = GlobalAvgPool2D<Float>()
    public var outputClassifier: Dense<Float>

    public init(classCount: Int = 1000, widthMultiplier: Float = 1.0) {
        inputConv = Conv2D<Float>(
            filterShape: (3, 3, 3, makeDivisible(filter: 32, widthMultiplier: widthMultiplier)),
            strides: (2, 2),
            padding: .valid)
        inputConvBatchNorm = BatchNorm(
            featureCount: makeDivisible(filter: 32, widthMultiplier: widthMultiplier))

        initialInvertedBottleneck = InitialInvertedBottleneckBlock(
            filters: (32, 16), widthMultiplier: widthMultiplier)

        residualBlockStack1 = InvertedBottleneckBlockStack(
            filters: (16, 24), widthMultiplier: widthMultiplier, blockCount: 2)
        residualBlockStack2 = InvertedBottleneckBlockStack(
            filters: (24, 32), widthMultiplier: widthMultiplier, blockCount: 3)
        residualBlockStack3 = InvertedBottleneckBlockStack(
            filters: (32, 64), widthMultiplier: widthMultiplier, blockCount: 4)
        residualBlockStack4 = InvertedBottleneckBlockStack(
            filters: (64, 96), widthMultiplier: widthMultiplier, blockCount: 3,
            initialStrides: (1, 1))
        residualBlockStack5 = InvertedBottleneckBlockStack(
            filters: (96, 160), widthMultiplier: widthMultiplier, blockCount: 3)

        invertedBottleneckBlock16 = InvertedBottleneckBlock(
            filters: (160, 320), widthMultiplier: widthMultiplier)

        var lastBlockFilterCount = makeDivisible(filter: 1280, widthMultiplier: widthMultiplier)
        if widthMultiplier < 1 {
            // paper: "One minor implementation difference, with [arxiv:1704.04861] is that for
            // multipliers less than one, we apply width multiplier to all layers except the very
            // last convolutional layer."
            lastBlockFilterCount = 1280
        }

        outputConv = Conv2D<Float>(
            filterShape: (
                1, 1,
                makeDivisible(filter: 320, widthMultiplier: widthMultiplier), lastBlockFilterCount
            ),
            strides: (1, 1),
            padding: .same)
        outputConvBatchNorm = BatchNorm(featureCount: lastBlockFilterCount)

        outputClassifier = Dense(
            inputSize: lastBlockFilterCount, outputSize: classCount,
            activation: softmax)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let convolved = relu6(input.sequenced(through: zeroPad, inputConv, inputConvBatchNorm))
        let initialConv = initialInvertedBottleneck(convolved)
        let backbone = initialConv.sequenced(
            through: residualBlockStack1, residualBlockStack2, residualBlockStack3,
            residualBlockStack4, residualBlockStack5)
        let output = relu6(outputConvBatchNorm(outputConv(invertedBottleneckBlock16(backbone))))
        return output.sequenced(through: avgPool, outputClassifier)
    }
}
