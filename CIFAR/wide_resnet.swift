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

struct BatchNormConv2DBlock: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var norm1: BatchNorm<Float>
    var conv1: Conv2D<Float>
    var norm2: BatchNorm<Float>
    var conv2: Conv2D<Float>

    init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .same
    ) {
        self.norm1 = BatchNorm(featureCount: filterShape.2)
        self.conv1 = Conv2D(filterShape: filterShape, strides: strides, padding: padding)
        self.norm2 = BatchNorm(featureCount: filterShape.3)
        self.conv2 = Conv2D(filterShape: filterShape, strides: (1, 1), padding: padding)
    }

    @differentiable
    func call(_ input: Input) -> Output {
        let firstLayer = conv1(relu(norm1(input)))
        return conv2(relu(norm2(firstLayer)))
    }
}

struct WideResnet16FirstBasicBlock: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var block1: BatchNormConv2DBlock
    var block2: BatchNormConv2DBlock
    var shortcut: Conv2D<Float>

    init(
        featureCounts: (Int, Int), 
        kernelSize: Int = 3, 
        widenFactor: Int = 1, 
        initialStride: (Int, Int) = (2, 2)
    ) {
        self.block1 = BatchNormConv2DBlock(
            filterShape: (kernelSize, kernelSize,
                featureCounts.0, featureCounts.1 * widenFactor),
            strides: initialStride)
        self.block2 = BatchNormConv2DBlock(
            filterShape: (kernelSize, kernelSize,
                featureCounts.1 * widenFactor, featureCounts.1 * widenFactor),
            strides: (1, 1))
        self.shortcut = Conv2D(
            filterShape: (1, 1, featureCounts.0, featureCounts.1 * widenFactor),
            strides: initialStride)
    }

    @differentiable
    func call(_ input: Input) -> Output {
        return input.sequenced(through: block1, block2) + shortcut(input)
    }
}

struct WideResnet16BasicBlock: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var block1: BatchNormConv2DBlock
    var block2: BatchNormConv2DBlock
    var shortcut: Conv2D<Float>

    init(
        featureCounts: (Int, Int), 
        kernelSize: Int = 3, 
        widenFactor: Int = 1, 
        initialStride: (Int, Int) = (2, 2)
    ) {
        self.block1 = BatchNormConv2DBlock(
            filterShape: (kernelSize, kernelSize,
                featureCounts.0 * widenFactor, featureCounts.1 * widenFactor),
            strides: initialStride)
        self.block2 = BatchNormConv2DBlock(
            filterShape: (kernelSize, kernelSize,
                featureCounts.1 * widenFactor, featureCounts.1 * widenFactor),
            strides: (1, 1))
        self.shortcut = Conv2D(
            filterShape: (1, 1, featureCounts.0 * widenFactor, featureCounts.1 * widenFactor),
            strides: initialStride)
    }

    @differentiable
    func call(_ input: Input) -> Output {
        return input.sequenced(through: block1, block2) + shortcut(input)
    }
}

struct WideResNet16: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var l1: Conv2D<Float>

    var l2 = WideResnet16FirstBasicBlock(featureCounts: (16, 16), widenFactor: 4,
        initialStride: (1, 1))
    var l3 = WideResnet16BasicBlock(featureCounts: (16, 32), widenFactor: 4)
    var l4 = WideResnet16BasicBlock(featureCounts: (32, 64), widenFactor: 4)
 
    var norm: BatchNorm<Float>
    var avgPool: AvgPool2D<Float>
    var flatten = Flatten<Float>()
    var classifier: Dense<Float>

    init() {
        self.l1 = Conv2D(filterShape: (3, 3, 3, 16), strides: (1, 1), padding: .same)
        self.norm = BatchNorm(featureCount: 64 * 4)
        self.avgPool = AvgPool2D(poolSize: (8, 8), strides: (8, 8))
        self.classifier = Dense(inputSize: 64 * 4, outputSize: 10)
    }

    @differentiable
    func call(_ input: Input) -> Output {
        let inputLayer = input.sequenced(through: l1, l2, l3, l4)
        let finalNorm = relu(norm(inputLayer))
        return finalNorm.sequenced(through: avgPool, flatten, classifier)
    }
}

struct WideResnet28FirstBasicBlock: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var block1: BatchNormConv2DBlock
    var block2: BatchNormConv2DBlock
    var block3: BatchNormConv2DBlock
    var block4: BatchNormConv2DBlock
    var shortcut: Conv2D<Float>

    init(
        featureCounts: (Int, Int), 
        kernelSize: Int = 3, 
        widenFactor: Int = 1, 
        initialStride: (Int, Int) = (2, 2)
    ) {
        self.block1 = BatchNormConv2DBlock(
            filterShape: (kernelSize, kernelSize, featureCounts.0, featureCounts.1 * widenFactor),
            strides: initialStride)
        self.block2 = BatchNormConv2DBlock(
            filterShape: (kernelSize, kernelSize,
                featureCounts.1 * widenFactor, featureCounts.1 * widenFactor),
            strides: (1, 1))
        self.block3 = BatchNormConv2DBlock(
            filterShape: (kernelSize, kernelSize,
                featureCounts.1 * widenFactor, featureCounts.1 * widenFactor),
            strides: (1, 1))
        self.block4 = BatchNormConv2DBlock(
            filterShape: (kernelSize, kernelSize,
                featureCounts.1 * widenFactor, featureCounts.1 * widenFactor),
            strides: (1, 1))
        self.shortcut = Conv2D(
            filterShape: (1, 1, featureCounts.0, featureCounts.1 * widenFactor),
            strides: initialStride)
    }

    @differentiable
    func call(_ input: Input) -> Output {
        return input.sequenced(through: block1, block2, block3, block4) + shortcut(input)
    }
}

struct WideResnet28BasicBlock: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var block1: BatchNormConv2DBlock
    var block2: BatchNormConv2DBlock
    var block3: BatchNormConv2DBlock
    var block4: BatchNormConv2DBlock
    var shortcut: Conv2D<Float>

    init(
        featureCounts: (Int, Int), 
        kernelSize: Int = 3, 
        widenFactor: Int = 1, 
        initialStride: (Int, Int) = (2, 2)
    ) {
        self.block1 = BatchNormConv2DBlock(
            filterShape: (kernelSize, kernelSize,
                featureCounts.0 * widenFactor, featureCounts.1 * widenFactor),
            strides: initialStride)
        self.block2 = BatchNormConv2DBlock(
            filterShape: (kernelSize, kernelSize,
                featureCounts.1 * widenFactor, featureCounts.1 * widenFactor),
            strides: (1, 1))
        self.block3 = BatchNormConv2DBlock(
            filterShape: (kernelSize, kernelSize,
                featureCounts.1 * widenFactor, featureCounts.1 * widenFactor),
            strides: (1, 1))
        self.block4 = BatchNormConv2DBlock(
            filterShape: (kernelSize, kernelSize,
                featureCounts.1 * widenFactor, featureCounts.1 * widenFactor),
            strides: (1, 1))
        self.shortcut = Conv2D(
            filterShape: (1, 1, featureCounts.0 * widenFactor, featureCounts.1 * widenFactor),
            strides: initialStride)
    }

    @differentiable
    func call(_ input: Input) -> Output {
        return input.sequenced(through: block1, block2, block3, block4) + shortcut(input)
    }
}

struct WideResNet28: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var l1: Conv2D<Float>

    var l2 = WideResnet28FirstBasicBlock(featureCounts: (16, 16), widenFactor: 10,
        initialStride: (1, 1))
    var l3 = WideResnet28BasicBlock(featureCounts: (16, 32), widenFactor: 10)
    var l4 = WideResnet28BasicBlock(featureCounts: (32, 64), widenFactor: 10)
 
    var norm: BatchNorm<Float>
    var avgPool: AvgPool2D<Float>
    var flatten = Flatten<Float>()
    var classifier: Dense<Float>

    init() {
        self.l1 = Conv2D(filterShape: (3, 3, 3, 16), strides: (1, 1), padding: .same)
        self.norm = BatchNorm(featureCount: 64 * 10)
        self.avgPool = AvgPool2D(poolSize: (8, 8), strides: (8, 8))
        self.classifier = Dense(inputSize: 64 * 10, outputSize: 10)
    }

    @differentiable
    func call(_ input: Input) -> Output {
        let inputLayer = input.sequenced(through: l1, l2, l3, l4)
        let finalNorm = relu(norm(inputLayer))
        return finalNorm.sequenced(through: avgPool, flatten, classifier)
    }
}
