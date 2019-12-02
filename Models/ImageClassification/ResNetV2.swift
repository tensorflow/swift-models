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

// Pre-activated Resnet (aka Resnet v2):
// Original Paper:
// "Identity Mappings in Deep Residual Networks"
// Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun
// https://arxiv.org/abs/1603.05027
// https://github.com/KaimingHe/resnet-1k-layers/

public struct Conv2DBatchNorm: Layer {
    public var conv: Conv2D<Float>
    public var norm: BatchNorm<Float>

    public init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid
    ) {
        self.conv = Conv2D(
          filterShape: filterShape, strides: strides, padding: padding, activation: identity)
        self.norm = BatchNorm(featureCount: filterShape.3)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: conv, norm)
    }
}

public struct BatchNormConv2D: Layer {
    public var norm: BatchNorm<Float>
    public var conv: Conv2D<Float>

    public init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid
    ) {
        self.norm = BatchNorm(featureCount: filterShape.2)
        self.conv = Conv2D(filterShape: filterShape, strides: strides, padding: padding)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return conv(relu(norm(input)))
    }
}

public struct PreActivatedResidualBasicBlock: Layer {
    public var layer1: BatchNormConv2D
    public var layer2: BatchNormConv2D

    public init(
        featureCounts: (Int, Int, Int, Int),
        kernelSize: Int = 3,
        strides: (Int, Int) = (1, 1)
    ) {
        self.layer1 = BatchNormConv2D(
            filterShape: (kernelSize, kernelSize, featureCounts.0, featureCounts.1),
            strides: strides,
            padding: .same)
        self.layer2 = BatchNormConv2D(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.3),
            strides: strides,
            padding: .same)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: layer1, layer2)
    }
}

public struct PreActivatedResidualBasicBlockShortcut: Layer {
    public var layer1: BatchNormConv2D
    public var layer2: BatchNormConv2D
    public var shortcut: Conv2D<Float>

    public init(featureCounts: (Int, Int, Int, Int), kernelSize: Int = 3) {
        self.layer1 = BatchNormConv2D(
            filterShape: (kernelSize, kernelSize, featureCounts.0, featureCounts.1),
            strides: (2, 2),
            padding: .same)
        self.layer2 = BatchNormConv2D(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            strides: (1, 1),
            padding: .same)
        self.shortcut = Conv2D(
            filterShape: (1, 1, featureCounts.0, featureCounts.3),
            strides: (2, 2),
            padding: .same)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: layer1, layer2) + shortcut(input)
    }
}

public struct PreActivatedResNet18: Layer {
    public var l1: Conv2DBatchNorm
    public var maxPool: MaxPool2D<Float>

    public var l2a = PreActivatedResidualBasicBlock(featureCounts: (64, 64, 64, 64))
    public var l2b = PreActivatedResidualBasicBlock(featureCounts: (64, 64, 64, 64))

    public var l3a = PreActivatedResidualBasicBlockShortcut(featureCounts: (64, 128, 128, 128))
    public var l3b = PreActivatedResidualBasicBlock(featureCounts: (128, 128, 128, 128))

    public var l4a = PreActivatedResidualBasicBlockShortcut(featureCounts: (128, 256, 256, 256))
    public var l4b = PreActivatedResidualBasicBlock(featureCounts: (256, 256, 256, 256))

    public var l5a = PreActivatedResidualBasicBlockShortcut(featureCounts: (256, 512, 512, 512))
    public var l5b = PreActivatedResidualBasicBlock(featureCounts: (512, 512, 512, 512))

    public var norm: BatchNorm<Float>
    public var avgPool: AvgPool2D<Float>
    public var flatten = Flatten<Float>()
    public var classifier: Dense<Float>

    public init(imageSize: Int, classCount: Int) {
        // default to the ImageNet case where imageSize == 224
        // Swift requires that all properties get initialized outside control flow
        l1 = Conv2DBatchNorm(filterShape: (7, 7, 3, 64), strides: (2, 2), padding: .same)
        maxPool = MaxPool2D(poolSize: (3, 3), strides: (2, 2))
        avgPool = AvgPool2D(poolSize: (7, 7), strides: (7, 7))
        if imageSize == 32 {
            l1 = Conv2DBatchNorm(filterShape: (3, 3, 3, 64), padding: .same)
            maxPool = MaxPool2D(poolSize: (1, 1), strides: (1, 1))  // no-op
            avgPool = AvgPool2D(poolSize: (4, 4), strides: (4, 4))
        }
        norm = BatchNorm(featureCount: 512)
        classifier = Dense(inputSize: 512, outputSize: classCount)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let inputLayer = input.sequenced(through: l1, maxPool)
        let level2 = inputLayer.sequenced(through: l2a, l2b)
        let level3 = level2.sequenced(through: l3a, l3b)
        let level4 = level3.sequenced(through: l4a, l4b)
        let level5 = level4.sequenced(through: l5a, l5b)
        let finalNorm = relu(norm(level5))
        return finalNorm.sequenced(through: avgPool, flatten, classifier)
    }
}

public struct PreActivatedResNet34: Layer {
    public var l1: Conv2DBatchNorm
    public var maxPool: MaxPool2D<Float>

    public var l2a = PreActivatedResidualBasicBlock(featureCounts: (64, 64, 64, 64))
    public var l2b = PreActivatedResidualBasicBlock(featureCounts: (64, 64, 64, 64))
    public var l2c = PreActivatedResidualBasicBlock(featureCounts: (64, 64, 64, 64))

    public var l3a = PreActivatedResidualBasicBlockShortcut(featureCounts: (64, 128, 128, 128))
    public var l3b = PreActivatedResidualBasicBlock(featureCounts: (128, 128, 128, 128))
    public var l3c = PreActivatedResidualBasicBlock(featureCounts: (128, 128, 128, 128))
    public var l3d = PreActivatedResidualBasicBlock(featureCounts: (128, 128, 128, 128))

    public var l4a = PreActivatedResidualBasicBlockShortcut(featureCounts: (128, 256, 256, 256))
    public var l4b = PreActivatedResidualBasicBlock(featureCounts: (256, 256, 256, 256))
    public var l4c = PreActivatedResidualBasicBlock(featureCounts: (256, 256, 256, 256))
    public var l4d = PreActivatedResidualBasicBlock(featureCounts: (256, 256, 256, 256))
    public var l4e = PreActivatedResidualBasicBlock(featureCounts: (256, 256, 256, 256))
    public var l4f = PreActivatedResidualBasicBlock(featureCounts: (256, 256, 256, 256))

    public var l5a = PreActivatedResidualBasicBlockShortcut(featureCounts: (256, 512, 512, 512))
    public var l5b = PreActivatedResidualBasicBlock(featureCounts: (512, 512, 512, 512))
    public var l5c = PreActivatedResidualBasicBlock(featureCounts: (512, 512, 512, 512))

    public var norm: BatchNorm<Float>
    public var avgPool: AvgPool2D<Float>
    public var flatten = Flatten<Float>()
    public var classifier: Dense<Float>

    public init(imageSize: Int, classCount: Int) {
        // default to the ImageNet case where imageSize == 224
        // Swift requires that all properties get initialized outside control flow
        l1 = Conv2DBatchNorm(filterShape: (7, 7, 3, 64), strides: (2, 2), padding: .same)
        maxPool = MaxPool2D(poolSize: (3, 3), strides: (2, 2))
        avgPool = AvgPool2D(poolSize: (7, 7), strides: (7, 7))
        if imageSize == 32 {
            l1 = Conv2DBatchNorm(filterShape: (3, 3, 3, 64), padding: .same)
            maxPool = MaxPool2D(poolSize: (1, 1), strides: (1, 1))  // no-op
            avgPool = AvgPool2D(poolSize: (4, 4), strides: (4, 4))
        }
        norm = BatchNorm(featureCount: 512)
        classifier = Dense(inputSize: 512, outputSize: classCount)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let inputLayer = input.sequenced(through: l1, maxPool)
        let level2 = inputLayer.sequenced(through: l2a, l2b, l2c)
        let level3 = level2.sequenced(through: l3a, l3b, l3c, l3d)
        let level4 = level3.sequenced(through: l4a, l4b, l4c, l4d, l4e, l4f)
        let level5 = level4.sequenced(through: l5a, l5b, l5c)
        let finalNorm = relu(norm(level5))
        return finalNorm.sequenced(through: avgPool, flatten, classifier)
    }
}
