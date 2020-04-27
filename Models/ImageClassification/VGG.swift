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
import ModelSupport

// Original Paper:
// "Very Deep Convolutional Networks for Large-Scale Image Recognition"
// Karen Simonyan, Andrew Zisserman
// https://arxiv.org/abs/1409.1556

public struct VGGBlock: Layer, EasyLayers {
    var blocks: [Self.Conv2D] = []
    var maxpool = MaxPool2D(poolSize: (2, 2), strides: (2, 2))

    public init(featureCounts: (Int, Int, Int, Int), blockCount: Int) {
        self.blocks = [Conv2D(filterShape: (3, 3, featureCounts.0, featureCounts.1),
            padding: .same,
            activation: relu)]
        for _ in 1..<blockCount {
            self.blocks += [Conv2D(filterShape: (3, 3, featureCounts.2, featureCounts.3),
                padding: .same,
                activation: relu)]
        }
    }

    @differentiable
    public func callAsFunction(_ input: TensorF) -> TensorF {
        return maxpool(blocks.differentiableReduce(input) { $1($0) })
    }
}

public struct VGG16: Layer, EasyLayers {
    var layer1: VGGBlock
    var layer2: VGGBlock
    var layer3: VGGBlock
    var layer4: VGGBlock
    var layer5: VGGBlock

    var flatten = Flatten()
    var dense1 = Dense(inputSize: 512 * 7 * 7, outputSize: 4096, activation: relu)
    var dense2 = Dense(inputSize: 4096, outputSize: 4096, activation: relu)
    var output: Self.Dense

    public init(classCount: Int = 1000) {
        layer1 = VGGBlock(featureCounts: (3, 64, 64, 64), blockCount: 2)
        layer2 = VGGBlock(featureCounts: (64, 128, 128, 128), blockCount: 2)
        layer3 = VGGBlock(featureCounts: (128, 256, 256, 256), blockCount: 3)
        layer4 = VGGBlock(featureCounts: (256, 512, 512, 512), blockCount: 3)
        layer5 = VGGBlock(featureCounts: (512, 512, 512, 512), blockCount: 3)
        output = Dense(inputSize: 4096, outputSize: classCount, activation: softmax)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let backbone = input.sequenced(through: layer1, layer2, layer3, layer4, layer5)
        return backbone.sequenced(through: flatten, dense1, dense2, output)
    }
}

public struct VGG19: Layer {
    var layer1: VGGBlock
    var layer2: VGGBlock
    var layer3: VGGBlock
    var layer4: VGGBlock
    var layer5: VGGBlock

    var flatten = Flatten<Float>()
    var dense1 = Dense<Float>(inputSize: 512 * 7 * 7, outputSize: 4096, activation: relu)
    var dense2 = Dense<Float>(inputSize: 4096, outputSize: 4096, activation: relu)
    var output: Dense<Float>

    public init(classCount: Int = 1000) {
        layer1 = VGGBlock(featureCounts: (3, 64, 64, 64), blockCount: 2)
        layer2 = VGGBlock(featureCounts: (64, 128, 128, 128), blockCount: 2)
        layer3 = VGGBlock(featureCounts: (128, 256, 256, 256), blockCount: 4)
        layer4 = VGGBlock(featureCounts: (256, 512, 512, 512), blockCount: 4)
        layer5 = VGGBlock(featureCounts: (512, 512, 512, 512), blockCount: 4)
        output = Dense(inputSize: 4096, outputSize: classCount, activation: softmax)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let backbone = input.sequenced(through: layer1, layer2, layer3, layer4, layer5)
        return backbone.sequenced(through: flatten, dense1, dense2, output)
    }
}
