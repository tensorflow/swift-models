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
// "Deep Residual Learning for Image Recognition"
// Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
// https://arxiv.org/abs/1512.03385
// using shortcut layer to connect BasicBlock layers (aka Option (B))
// see https://github.com/akamaster/pytorch_resnet_cifar10 for explanation

struct Conv2DBatchNorm: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var conv: Conv2D<Float>
    var norm: BatchNorm<Float>

    init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1)
    ) {
        self.conv = Conv2D(filterShape: filterShape, strides: strides, padding: .same)
        self.norm = BatchNorm(featureCount: filterShape.3)
    }

    @differentiable
    func call(_ input: Input) -> Output {
        return input.sequenced(through: conv, norm)
    }
}

struct BasicBlock: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var blocks: [Conv2DBatchNorm]
    var shortcut: Conv2DBatchNorm

    init(
        featureCounts: (Int, Int),
        kernelSize: Int = 3,
        strides: (Int, Int) = (2, 2),
        blockCount: Int = 3
    ) {
        self.blocks = [Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.0, featureCounts.1),
            strides: strides)]
        for _ in 2..<blockCount {
            self.blocks += [Conv2DBatchNorm(
                filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.1))]
        }
        self.shortcut = Conv2DBatchNorm(
            filterShape: (1, 1, featureCounts.0, featureCounts.1),
            strides: strides)
    }

    @differentiable
    func call(_ input: Input) -> Output {
        let blocksReduced = blocks.differentiableReduce(input) { last, layer in
            relu(layer(last))
        }
        return relu(blocksReduced + shortcut(input))
    }
}

struct ResNet: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var inputLayer = Conv2DBatchNorm(filterShape: (3, 3, 3, 16))

    var basicBlock1: BasicBlock
    var basicBlock2: BasicBlock
    var basicBlock3: BasicBlock

    init(blockCount: Int = 3) {
        basicBlock1 = BasicBlock(featureCounts:(16, 16), strides: (1, 1), blockCount: blockCount)
        basicBlock2 = BasicBlock(featureCounts:(16, 32), blockCount: blockCount)
        basicBlock3 = BasicBlock(featureCounts:(32, 64), blockCount: blockCount)
    }

    var averagePool = AvgPool2D<Float>(poolSize: (8, 8), strides: (8, 8))
    var flatten = Flatten<Float>()
    var classifier = Dense<Float>(inputSize: 64, outputSize: 10, activation: softmax)

    @differentiable
    func call(_ input: Input) -> Output {
        let tmp = relu(inputLayer(input))
        let convolved = tmp.sequenced(through: basicBlock1, basicBlock2, basicBlock3)
        return convolved.sequenced(through: averagePool, flatten, classifier)
    }
}

extension ResNet {
    enum Kind: Int {
        case resNet20 = 3
        case resNet32 = 5
        case resNet44 = 7
        case resNet56 = 9
        case resNet110 = 18
    }

    init(kind: Kind) {
        self.init(blockCount: kind.rawValue)
    }
}
