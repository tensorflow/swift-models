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
// "Gradient-Based Learning Applied to Document Recognition"
// Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner
// http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
//
// Note: this implementation connects all the feature maps in the second convolutional layer.
// Additionally, ReLU is used instead of sigmoid activations.
public struct LeNet: Layer, FloatModel {
    public var conv1 = Conv2D(filterShape: (5, 5, 1, 6), padding: .same, activation: relu)
    public var pool1 = AvgPool2D(poolSize: (2, 2), strides: (2, 2))
    public var conv2 = Conv2D(filterShape: (5, 5, 6, 16), activation: relu)
    public var pool2 = AvgPool2D(poolSize: (2, 2), strides: (2, 2))
    public var flatten = Flatten()
    public var fc1 = Dense(inputSize: 400, outputSize: 120, activation: relu)
    public var fc2 = Dense(inputSize: 120, outputSize: 84, activation: relu)
    public var fc3 = Dense(inputSize: 84, outputSize: 10)

    public init() {}

    @differentiable
    public func callAsFunction(_ input: TensorF) -> TensorF {
        let convolved = input.sequenced(through: conv1, pool1, conv2, pool2)
        return convolved.sequenced(through: flatten, fc1, fc2, fc3)
    }
}
