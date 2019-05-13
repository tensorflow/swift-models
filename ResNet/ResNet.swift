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

enum InputKind {
    case cifar
    case imagenet
}

struct ConvBN: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var conv: Conv2D<Float>
    var norm: BatchNorm<Float>

    init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid
    ) {
        self.conv = Conv2D(filterShape: filterShape, strides: strides, padding: padding)
        self.norm = BatchNorm(featureCount: filterShape.3)
    }

    @differentiable
    func call(_ input: Input) -> Output {
        return input.sequenced(through: conv, norm)
    }
}

struct ResidualBasicBlockShortcut: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var layer1: ConvBN
    var layer2: ConvBN
    var shortcut: ConvBN

    init(featureCounts: (Int, Int, Int, Int), kernelSize: Int = 3) {
        self.layer1 = ConvBN(
            filterShape: (kernelSize, kernelSize, featureCounts.0, featureCounts.1),
            strides: (2, 2),
            padding: .same)
        self.layer2 = ConvBN(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            strides: (1, 1),
            padding: .same)
        self.shortcut = ConvBN(
            filterShape: (1, 1, featureCounts.0, featureCounts.3),
            strides: (2, 2),
            padding: .same)
    }

    @differentiable
    func call(_ input: Input) -> Output {
        return layer2(relu(layer1(input))) + shortcut(input)
    }
}

struct ResidualBasicBlock: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var layer1: ConvBN
    var layer2: ConvBN

    init(
        featureCounts: (Int, Int, Int, Int),
        kernelSize: Int = 3,
        strides: (Int, Int) = (1, 1)
    ) {
        self.layer1 = ConvBN(
            filterShape: (kernelSize, kernelSize, featureCounts.0, featureCounts.1),
            strides: strides,
            padding: .same)
        self.layer2 = ConvBN(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.3),
            strides: strides,
            padding: .same)
    }

    @differentiable
    func call(_ input: Input) -> Output {
        return layer2(relu(layer1(input)))
    }
}

struct ResidualBasicBlockStack: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var blocks: [ResidualBasicBlock] = []
    init(featureCounts: (Int, Int, Int, Int), kernelSize: Int = 3, blockCount: Int) {
        for _ in 1..<blockCount {
            blocks += [ResidualBasicBlock(featureCounts: featureCounts, kernelSize: kernelSize)]
        }
    }

    @differentiable
    func call(_ input: Input) -> Output {
        let blocksReduced = blocks.differentiableReduce(input) { last, layer in
            layer(last)
        }
        return blocksReduced
    }
}

struct ResidualConvBlock: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var layer1: ConvBN
    var layer2: ConvBN
    var layer3: ConvBN
    var shortcut: ConvBN

    init(
        featureCounts: (Int, Int, Int, Int),
        kernelSize: Int = 3,
        strides: (Int, Int) = (2, 2)
    ) {
        self.layer1 = ConvBN(
            filterShape: (1, 1, featureCounts.0, featureCounts.1),
            strides: strides)
        self.layer2 = ConvBN(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            padding: .same)
        self.layer3 = ConvBN(filterShape: (1, 1, featureCounts.2, featureCounts.3))
        self.shortcut = ConvBN(
            filterShape: (1, 1, featureCounts.0, featureCounts.3),
            strides: strides,
            padding: .same)
    }

    @differentiable
    func call(_ input: Input) -> Output {
        let tmp = relu(layer2(relu(layer1(input))))
        return relu(layer3(tmp) + shortcut(input))
    }
}

struct ResidualIdentityBlock: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var layer1: ConvBN
    var layer2: ConvBN
    var layer3: ConvBN

    init(featureCounts: (Int, Int, Int, Int), kernelSize: Int = 3) {
        self.layer1 = ConvBN(filterShape: (1, 1, featureCounts.0, featureCounts.1))
        self.layer2 = ConvBN(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            padding: .same)
        self.layer3 = ConvBN(filterShape: (1, 1, featureCounts.2, featureCounts.3))
    }

    @differentiable
    func call(_ input: Input) -> Output {
        let tmp = relu(layer2(relu(layer1(input))))
        return relu(layer3(tmp) + input)
    }
}

struct ResidualIdentityBlockStack: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var blocks: [ResidualIdentityBlock] = []
    init(featureCounts: (Int, Int, Int, Int), kernelSize: Int = 3, blockCount: Int) {
        for _ in 1..<blockCount {
            blocks += [ResidualIdentityBlock(featureCounts: featureCounts, kernelSize: kernelSize)]
        }
    }

    @differentiable
    func call(_ input: Input) -> Output {
        let blocksReduced = blocks.differentiableReduce(input) { last, layer in
            layer(last)
        }
        return blocksReduced
    }
}

struct ResNetBasic: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var l1: ConvBN
    var maxPool: MaxPool2D<Float>

    var l2a = ResidualBasicBlock(featureCounts: (64, 64, 64, 64))
    var l2b: ResidualBasicBlockStack

    var l3a = ResidualBasicBlockShortcut(featureCounts: (64, 128, 128, 128))
    var l3b: ResidualBasicBlockStack

    var l4a = ResidualBasicBlockShortcut(featureCounts: (128, 256, 256, 256))
    var l4b: ResidualBasicBlockStack

    var l5a = ResidualBasicBlockShortcut(featureCounts: (256, 512, 512, 512))
    var l5b: ResidualBasicBlockStack

    var avgPool: AvgPool2D<Float>
    var flatten = Flatten<Float>()
    var classifier: Dense<Float>

    init(input: InputKind, layerBlockCounts: (Int, Int, Int, Int)) {
        switch input {
        case .imagenet:
            l1 = ConvBN(filterShape: (7, 7, 3, 64), strides: (2, 2), padding: .same)
            maxPool = MaxPool2D(poolSize: (3, 3), strides: (2, 2))
            avgPool = AvgPool2D(poolSize: (7, 7), strides: (7, 7))
            classifier = Dense(inputSize: 512, outputSize: 1000)
        case .cifar:
            l1 = ConvBN(filterShape: (3, 3, 3, 64), padding: .same)
            maxPool = MaxPool2D(poolSize: (1, 1), strides: (1, 1)) // no-op
            avgPool = AvgPool2D(poolSize: (4, 4), strides: (4, 4))
            classifier = Dense(inputSize: 512, outputSize: 10)
        }

        l2b = ResidualBasicBlockStack(featureCounts: (64, 64, 64, 64),
            blockCount: layerBlockCounts.0)
        l3b = ResidualBasicBlockStack(featureCounts: (128, 128, 128, 128),
            blockCount: layerBlockCounts.1)
        l4b = ResidualBasicBlockStack(featureCounts: (256, 256, 256, 256),
            blockCount: layerBlockCounts.2)
        l5b = ResidualBasicBlockStack(featureCounts: (512, 512, 512, 512),
            blockCount: layerBlockCounts.3)
    }

    @differentiable
    func call(_ input: Input) -> Output {
        let inputLayer = maxPool(relu(l1(input)))
        let level2 = inputLayer.sequenced(through: l2a, l2b)
        let level3 = level2.sequenced(through: l3a, l3b)
        let level4 = level3.sequenced(through: l4a, l4b)
        let level5 = level4.sequenced(through: l5a, l5b)
        return level5.sequenced(through: avgPool, flatten, classifier)
    }
}

extension ResNetBasic {
    enum Kind {
        case resNet18
        case resNet34
    }

    init(kind: Kind, type: InputKind) {
        switch kind {
        case .resNet18:
            self.init(input: type, layerBlockCounts: (2, 2, 2, 2))
        case .resNet34:
            self.init(input: type, layerBlockCounts: (3, 4, 6, 3))
        }
    }
}

struct ResNet: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var l1: ConvBN
    var maxPool: MaxPool2D<Float>

    var l2a = ResidualConvBlock(featureCounts: (64, 64, 64, 256), strides: (1, 1))
    var l2b: ResidualIdentityBlockStack

    var l3a = ResidualConvBlock(featureCounts: (256, 128, 128, 512))
    var l3b: ResidualIdentityBlockStack

    var l4a = ResidualConvBlock(featureCounts: (512, 256, 256, 1024))
    var l4b: ResidualIdentityBlockStack

    var l5a = ResidualConvBlock(featureCounts: (1024, 512, 512, 2048))
    var l5b: ResidualIdentityBlockStack

    var avgPool: AvgPool2D<Float>
    var flatten = Flatten<Float>()
    var classifier: Dense<Float>

    init(input: InputKind, layerBlockCounts: (Int, Int, Int, Int)) {
        switch input {
        case .imagenet:
            l1 = ConvBN(filterShape: (7, 7, 3, 64), strides: (2, 2), padding: .same)
            maxPool = MaxPool2D(poolSize: (3, 3), strides: (2, 2))
            avgPool = AvgPool2D(poolSize: (7, 7), strides: (7, 7))
            classifier = Dense(inputSize: 2048, outputSize: 1000)
        case .cifar:
            l1 = ConvBN(filterShape: (3, 3, 3, 64), padding: .same)
            maxPool = MaxPool2D(poolSize: (1, 1), strides: (1, 1)) // no-op
            avgPool = AvgPool2D(poolSize: (4, 4), strides: (4, 4))
            classifier = Dense(inputSize: 2048, outputSize: 10)
        }

        l2b = ResidualIdentityBlockStack(featureCounts: (256, 64, 64, 256),
            blockCount: layerBlockCounts.0)
        l3b = ResidualIdentityBlockStack(featureCounts: (512, 128, 128, 512),
            blockCount: layerBlockCounts.1)
        l4b = ResidualIdentityBlockStack(featureCounts: (1024, 256, 256, 1024),
            blockCount: layerBlockCounts.2)
        l5b = ResidualIdentityBlockStack(featureCounts: (2048, 512, 512, 2048),
            blockCount: layerBlockCounts.3)
    }

    @differentiable
    func call(_ input: Input) -> Output {
        let inputLayer = maxPool(relu(l1(input)))
        let level2 = inputLayer.sequenced(through: l2a, l2b)
        let level3 = level2.sequenced(through: l3a, l3b)
        let level4 = level3.sequenced(through: l4a, l4b)
        let level5 = level4.sequenced(through: l5a, l5b)
        return level5.sequenced(through: avgPool, flatten, classifier)
    }
}

extension ResNet {
    enum Kind {
        case resNet50
        case resNet101
        case resNet152
    }

    init(kind: Kind, type: InputKind) {
        switch kind {
        case .resNet50:
            self.init(input: type, layerBlockCounts: (3, 4, 6, 3))
        case .resNet101:
            self.init(input: type, layerBlockCounts: (3, 4, 23, 3))
        case .resNet152:
            self.init(input: type, layerBlockCounts: (3, 8, 36, 3))
        }
    }
}
