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
// Densely Connected Convolutional Networks
// Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
// https://arxiv.org/pdf/1608.06993.pdf

public struct DenseNet121: Layer {
    public var conv = Conv(
        filterSize: 7,
        stride: 2,
        inputFilterCount: 3,
        outputFilterCount: 64
    )
    public var maxpool = MaxPool2D<Float>(
        poolSize: (3, 3),
        strides: (2, 2),
        padding: .same
    )
    public var denseBlock1 = DenseBlock(repetitionCount: 6, inputFilterCount: 64)
    public var transitionLayer1 = TransitionLayer(inputFilterCount: 256)
    public var denseBlock2 = DenseBlock(repetitionCount: 12, inputFilterCount: 128)
    public var transitionLayer2 = TransitionLayer(inputFilterCount: 512)
    public var denseBlock3 = DenseBlock(repetitionCount: 24, inputFilterCount: 256)
    public var transitionLayer3 = TransitionLayer(inputFilterCount: 1024)
    public var denseBlock4 = DenseBlock(repetitionCount: 16, inputFilterCount: 512)
    public var globalAvgPool = GlobalAvgPool2D<Float>()
    public var dense: Dense<Float>

    public init(classCount: Int) {
        dense = Dense(inputSize: 1024, outputSize: classCount)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let inputLayer = input.sequenced(through: conv, maxpool)
        let level1 = inputLayer.sequenced(through: denseBlock1, transitionLayer1)
        let level2 = level1.sequenced(through: denseBlock2, transitionLayer2)
        let level3 = level2.sequenced(through: denseBlock3, transitionLayer3)
        let output = level3.sequenced(through: denseBlock4, globalAvgPool, dense)
        return output
    }
}

public struct DenseNet169: Layer {
    public var conv = Conv(
        filterSize: 7,
        stride: 2,
        inputFilterCount: 3,
        outputFilterCount: 64
    )
    public var maxpool = MaxPool2D<Float>(
        poolSize: (3, 3),
        strides: (2, 2),
        padding: .same
    )
    public var denseBlock1 = DenseBlock(repetitionCount: 6, inputFilterCount: 64)
    public var transitionLayer1 = TransitionLayer(inputFilterCount: 256)
    public var denseBlock2 = DenseBlock(repetitionCount: 12, inputFilterCount: 128)
    public var transitionLayer2 = TransitionLayer(inputFilterCount: 512)
    public var denseBlock3 = DenseBlock(repetitionCount: 32, inputFilterCount: 256)
    public var transitionLayer3 = TransitionLayer(inputFilterCount: 1280)
    public var denseBlock4 = DenseBlock(repetitionCount: 32, inputFilterCount: 640)
    public var globalAvgPool = GlobalAvgPool2D<Float>()
    public var dense: Dense<Float>

    public init(classCount: Int) {
        dense = Dense(inputSize: 1664, outputSize: classCount)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let inputLayer = input.sequenced(through: conv, maxpool)
        let level1 = inputLayer.sequenced(through: denseBlock1, transitionLayer1)
        let level2 = level1.sequenced(through: denseBlock2, transitionLayer2)
        let level3 = level2.sequenced(through: denseBlock3, transitionLayer3)
        let output = level3.sequenced(through: denseBlock4, globalAvgPool, dense)
        return output
    }
}

public struct DenseNet210: Layer {
    public var conv = Conv(
        filterSize: 7,
        stride: 2,
        inputFilterCount: 3,
        outputFilterCount: 64
    )
    public var maxpool = MaxPool2D<Float>(
        poolSize: (3, 3),
        strides: (2, 2),
        padding: .same
    )
    public var denseBlock1 = DenseBlock(repetitionCount: 6, inputFilterCount: 64)
    public var transitionLayer1 = TransitionLayer(inputFilterCount: 256)
    public var denseBlock2 = DenseBlock(repetitionCount: 12, inputFilterCount: 128)
    public var transitionLayer2 = TransitionLayer(inputFilterCount: 512)
    public var denseBlock3 = DenseBlock(repetitionCount: 48, inputFilterCount: 256)
    public var transitionLayer3 = TransitionLayer(inputFilterCount: 1792)
    public var denseBlock4 = DenseBlock(repetitionCount: 32, inputFilterCount: 896)
    public var globalAvgPool = GlobalAvgPool2D<Float>()
    public var dense: Dense<Float>

    public init(classCount: Int) {
        dense = Dense(inputSize: 1920, outputSize: classCount)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let inputLayer = input.sequenced(through: conv, maxpool)
        let level1 = inputLayer.sequenced(through: denseBlock1, transitionLayer1)
        let level2 = level1.sequenced(through: denseBlock2, transitionLayer2)
        let level3 = level2.sequenced(through: denseBlock3, transitionLayer3)
        let output = level3.sequenced(through: denseBlock4, globalAvgPool, dense)
        return output
    }
}


public struct Conv: Layer {
    public var batchNorm: BatchNorm<Float>
    public var conv: Conv2D<Float>

    public init(
        filterSize: Int,
        stride: Int = 1,
        inputFilterCount: Int,
        outputFilterCount: Int
    ) {
        batchNorm = BatchNorm(featureCount: inputFilterCount)
        conv = Conv2D(
            filterShape: (filterSize, filterSize, inputFilterCount, outputFilterCount),
            strides: (stride, stride),
            padding: .same
        )
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        conv(relu(batchNorm(input)))
    }
}

/// A pair of a 1x1 `Conv` layer and a 3x3 `Conv` layer.
public struct ConvPair: Layer {
    public var conv1x1: Conv
    public var conv3x3: Conv

    public init(inputFilterCount: Int, growthRate: Int) {
        conv1x1 = Conv(
            filterSize: 1,
            inputFilterCount: inputFilterCount,
            outputFilterCount: inputFilterCount * 2
        )
        conv3x3 = Conv(
            filterSize: 3,
            inputFilterCount: inputFilterCount * 2,
            outputFilterCount: growthRate
        )
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let conv1Output = conv1x1(input)
        let conv3Output = conv3x3(conv1Output)
        return conv3Output.concatenated(with: input, alongAxis: -1)
    }
}

public struct DenseBlock: Layer {
    public var pairs: [ConvPair] = []

    public init(repetitionCount: Int, growthRate: Int = 32, inputFilterCount: Int) {
        for i in 0..<repetitionCount {
            let filterCount = inputFilterCount + i * growthRate
            pairs.append(ConvPair(inputFilterCount: filterCount, growthRate: growthRate))
        }
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        pairs.differentiableReduce(input) { last, layer in
            layer(last)
        }
    }
}

public struct TransitionLayer: Layer {
    public var conv: Conv
    public var pool: AvgPool2D<Float>

    public init(inputFilterCount: Int) {
        conv = Conv(
            filterSize: 1,
            inputFilterCount: inputFilterCount,
            outputFilterCount: inputFilterCount / 2
        )
        pool = AvgPool2D(poolSize: (2, 2), strides: (2, 2), padding: .same)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        input.sequenced(through: conv, pool)
    }
}

