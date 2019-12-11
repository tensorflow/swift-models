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
// "Very Deep Convolutional Networks for Large-Scale Image Recognition"
// Karen Simonyan, Andrew Zisserman
// https://arxiv.org/abs/1409.1556

public struct VGG16: Layer {
    public typealias Input = Tensor<Float>
    public typealias Output = Tensor<Float>

    var conv1a = Conv2D<Float>(filterShape: (3, 3, 3, 64), padding: .same, activation: relu)
    var conv1b = Conv2D<Float>(filterShape: (3, 3, 64, 64), padding: .same, activation: relu)
    var pool1 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))

    var conv2a = Conv2D<Float>(filterShape: (3, 3, 64, 128), padding: .same, activation: relu)
    var conv2b = Conv2D<Float>(filterShape: (3, 3, 128, 128), padding: .same, activation: relu)
    var pool2 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))

    var conv3a = Conv2D<Float>(filterShape: (3, 3, 128, 256), padding: .same, activation: relu)
    var conv3b = Conv2D<Float>(filterShape: (3, 3, 256, 256), padding: .same, activation: relu)
    var conv3c = Conv2D<Float>(filterShape: (3, 3, 256, 256), padding: .same, activation: relu)
    var pool3 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))

    var conv4a = Conv2D<Float>(filterShape: (3, 3, 256, 512), padding: .same, activation: relu)
    var conv4b = Conv2D<Float>(filterShape: (3, 3, 512, 512), padding: .same, activation: relu)
    var conv4c = Conv2D<Float>(filterShape: (3, 3, 512, 512), padding: .same, activation: relu)
    var pool4 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))

    var conv5a = Conv2D<Float>(filterShape: (3, 3, 512, 512), padding: .same, activation: relu)
    var conv5b = Conv2D<Float>(filterShape: (3, 3, 512, 512), padding: .same, activation: relu)
    var conv5c = Conv2D<Float>(filterShape: (3, 3, 512, 512), padding: .same, activation: relu)
    var pool5 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))

    var flatten = Flatten<Float>()
    var dense1 = Dense<Float>(inputSize: 512 * 7 * 7, outputSize: 4096, activation: relu)
    var dense2 = Dense<Float>(inputSize: 4096, outputSize: 4096, activation: relu)
    var dense3 = Dense<Float>(inputSize: 4096, outputSize: 1000, activation: softmax)

    public init() {}

    @differentiable
    public func callAsFunction(_ input: Input) -> Output {
        let conv1 = input.sequenced(through: conv1a, conv1b, pool1)
        let conv2 = conv1.sequenced(through: conv2a, conv2b, pool2)
        let conv3 = conv2.sequenced(through: conv3a, conv3b, conv3c, pool3)
        let conv4 = conv3.sequenced(through: conv4a, conv4b, conv4c, pool4)
        let conv5 = conv4.sequenced(through: conv5a, conv5b, conv5c, pool5)
        return conv5.sequenced(through: flatten, dense1, dense2, dense3)
    }
}
