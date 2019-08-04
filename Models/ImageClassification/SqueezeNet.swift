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
// Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally and Kurt Keutzer
// https://arxiv.org/pdf/1602.07360.pdf

public struct Fire: Layer {

    public var squeeze: Conv2D<Float>
    public var expand1: Conv2D<Float>
    public var expand3: Conv2D<Float>

    public init(inputFilterCount:Int, s1x1: Int, e1x1: Int, e3x3: Int) {
        squeeze = Conv2D(filterShape: (1, 1, inputFilterCount, s1x1), activation: relu)
        expand1 = Conv2D(filterShape: (1, 1, s1x1, e1x1), activation: relu)
        expand3 = Conv2D(filterShape: (3, 3, s1x1, e3x3), padding: .same, activation: relu)

    }
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let squeezed = squeeze(input)
        let expanded1 = expand1(squeezed)
        let expanded3 = expand3(squeezed)
        return expanded1.concatenated(with: expanded3, alongAxis: -1)
    }

}

public struct SqueezeNet: Layer {

    public var conv1: Conv2D<Float>
    public var maxpool1: MaxPool2D<Float>
    public var fire2: Fire
    public var fire3: Fire
    public var fire4: Fire
    public var maxpool4: MaxPool2D<Float>
    public var fire5: Fire
    public var fire6: Fire
    public var fire7: Fire
    public var fire8: Fire
    public var maxpool8: MaxPool2D<Float>
    public var fire9: Fire
    public var conv10: Conv2D<Float>
    public var avgpool10: AvgPool2D<Float>

    public init() {
        // Convolved-1
        conv1 = Conv2D(filterShape: (7, 7, 3, 96), strides: (2, 2), padding:.same)
        maxpool1 = MaxPool2D(poolSize: (3, 3), strides: (2, 2))

        // Fired-1
        fire2 = Fire(inputFilterCount: 96, s1x1: 16, e1x1: 64, e3x3: 64)
        fire3 = Fire(inputFilterCount: 128,s1x1: 16, e1x1: 64, e3x3: 64)
        fire4 = Fire(inputFilterCount: 128,s1x1: 32, e1x1: 128, e3x3: 128)
        maxpool4 = MaxPool2D(poolSize: (3, 3), strides: (2, 2))
        fire5 = Fire(inputFilterCount: 256,s1x1: 32, e1x1: 128, e3x3: 128)
        fire6 = Fire(inputFilterCount: 256,s1x1: 48, e1x1: 192, e3x3: 192)
        
        // Fired-2
        fire7 = Fire(inputFilterCount: 384,s1x1: 48, e1x1: 192, e3x3: 192)
        fire8 = Fire(inputFilterCount: 384,s1x1: 64, e1x1: 256, e3x3: 256)
        maxpool8 = MaxPool2D(poolSize: (3, 3), strides: (2, 2))
        fire9 = Fire(inputFilterCount: 512,s1x1: 64, e1x1: 256, e3x3: 256)

        // Convolved-2
        conv10 = Conv2D(filterShape: (1, 1, 512, 1000), strides: (1, 1))
        avgpool10 = AvgPool2D(poolSize: (13, 13), strides: (1, 1))
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let convolved1 = input.sequenced(through: conv1, maxpool1)
        let fired1 = convolved1.sequenced(through: fire2, fire3, fire4, maxpool4, fire5, fire6)
        let fired2 = (fired1.sequenced(through: fire7, fire8, maxpool8, fire9)).droppingOut(probability: 0.5)
        let convolved2 = fired2.sequenced(through: conv10, avgpool10)
        return convolved2
    }
}
