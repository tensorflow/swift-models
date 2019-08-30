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
// SqueezeNet: AlexNet Level Accuracy with 50X Fewer Parameters
// Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally
// and Kurt Keutzer
// https://arxiv.org/pdf/1602.07360.pdf

public struct Fire: Layer {
    public var squeeze: Conv2D<Float>
    public var expand1: Conv2D<Float>
    public var expand3: Conv2D<Float>

    public init(
        inputFilterCount: Int,
        squeezeFilterCount: Int,
        expand1FilterCount: Int,
        expand3FilterCount: Int
    ) {
        squeeze = Conv2D(
            filterShape: (1, 1, inputFilterCount, squeezeFilterCount),
            activation: relu)
        expand1 = Conv2D(
            filterShape: (1, 1, squeezeFilterCount, expand1FilterCount),
            activation: relu)
        expand3 = Conv2D(
            filterShape: (3, 3, squeezeFilterCount, expand3FilterCount),
            padding: .same,
            activation: relu)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let squeezed = squeeze(input)
        let expanded1 = expand1(squeezed)
        let expanded3 = expand3(squeezed)
        return expanded1.concatenated(with: expanded3, alongAxis: -1)
    }
}

public struct SqueezeNetV1_0: Layer {
    public var conv1 = Conv2D<Float>(filterShape: (7, 7, 3, 96), strides: (2, 2), padding: .same,
                                     activation: relu)
    public var maxPool1 = MaxPool2D<Float>(poolSize: (3, 3), strides: (2, 2))
    public var fire2 = Fire(
        inputFilterCount: 96,
        squeezeFilterCount: 16,
        expand1FilterCount: 64,
        expand3FilterCount: 64)
    public var fire3 = Fire(
        inputFilterCount: 128,
        squeezeFilterCount: 16,
        expand1FilterCount: 64,
        expand3FilterCount: 64)
    public var fire4 = Fire(
        inputFilterCount: 128,
        squeezeFilterCount: 32,
        expand1FilterCount: 128,
        expand3FilterCount: 128)
    public var maxPool4 = MaxPool2D<Float>(poolSize: (3, 3), strides: (2, 2))
    public var fire5 = Fire(
        inputFilterCount: 256,
        squeezeFilterCount: 32,
        expand1FilterCount: 128,
        expand3FilterCount: 128)
    public var fire6 = Fire(
        inputFilterCount: 256,
        squeezeFilterCount: 48,
        expand1FilterCount: 192,
        expand3FilterCount: 192)
    public var fire7 = Fire(
        inputFilterCount: 384,
        squeezeFilterCount: 48,
        expand1FilterCount: 192,
        expand3FilterCount: 192)
    public var fire8 = Fire(
        inputFilterCount: 384,
        squeezeFilterCount: 64,
        expand1FilterCount: 256,
        expand3FilterCount: 256)
    public var maxPool8 = MaxPool2D<Float>(poolSize: (3, 3), strides: (2, 2))
    public var fire9 = Fire(
        inputFilterCount: 512,
        squeezeFilterCount: 64,
        expand1FilterCount: 256,
        expand3FilterCount: 256)
    public var conv10: Conv2D<Float>
    public var avgPool10 = AvgPool2D<Float>(poolSize: (13, 13), strides: (1, 1))
    public var dropout = Dropout<Float>(probability: 0.5)

    public init(classCount: Int) {
        conv10 = Conv2D(filterShape: (1, 1, 512, classCount), strides: (1, 1), activation: relu)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let convolved1 = input.sequenced(through: conv1, maxPool1)
        let fired1 = convolved1.sequenced(through: fire2, fire3, fire4, maxPool4, fire5, fire6)
        let fired2 = fired1.sequenced(through: fire7, fire8, maxPool8, fire9)
        let convolved2 = fired2.sequenced(through: dropout, conv10, avgPool10)
        return convolved2
    }
}

public struct SqueezeNetV1_1: Layer {
    public var conv1 = Conv2D<Float>(filterShape: (3, 3, 3, 64), strides: (2, 2), padding: .same,
                                     activation: relu)
    public var maxPool1 = MaxPool2D<Float>(poolSize: (3, 3), strides: (2, 2))
    public var fire2 = Fire(
        inputFilterCount: 64,
        squeezeFilterCount: 16,
        expand1FilterCount: 64,
        expand3FilterCount: 64)
    public var fire3 = Fire(
        inputFilterCount: 128,
        squeezeFilterCount: 16,
        expand1FilterCount: 64,
        expand3FilterCount: 64)
    public var maxPool3 = MaxPool2D<Float>(poolSize: (3, 3), strides: (2, 2))
    public var fire4 = Fire(
        inputFilterCount: 128,
        squeezeFilterCount: 32,
        expand1FilterCount: 128,
        expand3FilterCount: 128)
    public var fire5 = Fire(
        inputFilterCount: 256,
        squeezeFilterCount: 32,
        expand1FilterCount: 128,
        expand3FilterCount: 128)
    public var maxPool5 = MaxPool2D<Float>(poolSize: (3, 3), strides: (2, 2))
    public var fire6 = Fire(
        inputFilterCount: 256,
        squeezeFilterCount: 48,
        expand1FilterCount: 192,
        expand3FilterCount: 192)
    public var fire7 = Fire(
        inputFilterCount: 384,
        squeezeFilterCount: 48,
        expand1FilterCount: 192,
        expand3FilterCount: 192)
    public var fire8 = Fire(
        inputFilterCount: 384,
        squeezeFilterCount: 64,
        expand1FilterCount: 256,
        expand3FilterCount: 256)
    public var fire9 = Fire(
        inputFilterCount: 512,
        squeezeFilterCount: 64,
        expand1FilterCount: 256,
        expand3FilterCount: 256)
    public var conv10: Conv2D<Float>
    public var avgPool10 = AvgPool2D<Float>(poolSize: (13, 13), strides: (1, 1))
    public var dropout = Dropout<Float>(probability: 0.5)

    public init(classCount: Int) {
        conv10 = Conv2D(filterShape: (1, 1, 512, classCount), strides: (1, 1), activation: relu)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let convolved1 = input.sequenced(through: conv1, maxPool1)
        let fired1 = convolved1.sequenced(through: fire2, fire3, maxPool3, fire4, fire5)
        let fired2 = fired1.sequenced(through: maxPool5, fire6, fire7, fire8, fire9)
        let convolved2 = fired2.sequenced(through: dropout, conv10, avgPool10)
        return convolved2
    }
}
