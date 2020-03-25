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

public struct ResNetGenerator<NormalizationType: FeatureChannelInitializable>: Layer where NormalizationType.TangentVector.VectorSpaceScalar == Float, NormalizationType.Input == Tensorf, NormalizationType.Output == Tensorf {
    var conv1: Conv2D<Float>
    var norm1: NormalizationType

    var conv2: Conv2D<Float>
    var norm2: NormalizationType

    var conv3: Conv2D<Float>
    var norm3: NormalizationType

    var resblocks: [ResNetBlock<NormalizationType>]

    var upConv1: TransposedConv2D<Float>
    var upNorm1: NormalizationType

    var upConv2: TransposedConv2D<Float>
    var upNorm2: NormalizationType

    var lastConv: Conv2D<Float>

    public init(inputChannels: Int,
                outputChannels: Int,
                blocks: Int,
                ngf: Int,
                normalization: NormalizationType.Type,
                useDropout: Bool = false) {
        norm1 = NormalizationType(featureCount: ngf)
        let useBias = norm1 is InstanceNorm2D<Float>

        let filterInit: (TensorShape) -> Tensorf = { Tensorf(randomNormal: $0, standardDeviation: Tensorf(0.02)) }
        let biasInit: (TensorShape) -> Tensorf = useBias ? filterInit : zeros()

        conv1 = Conv2D(filterShape: (7, 7, inputChannels, ngf),
                       strides: (1, 1),
                       filterInitializer: filterInit,
                       biasInitializer: biasInit)

        var mult = 1

        conv2 = Conv2D(filterShape: (3, 3, ngf * mult, ngf * mult * 2),
                       strides: (2, 2),
                       padding: .same,
                       filterInitializer: filterInit,
                       biasInitializer: biasInit)
        norm2 = NormalizationType(featureCount: ngf * mult * 2)

        mult = 2

        conv3 = Conv2D(filterShape: (3, 3, ngf * mult, ngf * mult * 2),
                       strides: (2, 2),
                       padding: .same,
                       filterInitializer: filterInit,
                       biasInitializer: biasInit)
        norm3 = NormalizationType(featureCount: ngf * mult * 2)

        mult = 4

        resblocks = (0 ..< blocks).map { _ in
            ResNetBlock(channels: ngf * mult,
                        paddingMode: .reflect,
                        normalization: normalization,
                        useDropOut: useDropout,
                        filterInit: filterInit,
                        biasInit: biasInit)
        }

        mult = 4

        upConv1 = TransposedConv2D(filterShape: (3, 3, ngf * mult / 2, ngf * mult),
                                   strides: (2, 2),
                                   padding: .same,
                                   filterInitializer: filterInit,
                                   biasInitializer: biasInit)
        upNorm1 = NormalizationType(featureCount: ngf * mult / 2)

        mult = 2

        upConv2 = TransposedConv2D(filterShape: (3, 3, ngf * mult / 2, ngf * mult),
                                   strides: (2, 2),
                                   padding: .same,
                                   filterInitializer: filterInit,
                                   biasInitializer: biasInit)
        upNorm2 = NormalizationType(featureCount: ngf * mult / 2)

        lastConv = Conv2D(filterShape: (7, 7, ngf, outputChannels),
                          padding: .same,
                          filterInitializer: filterInit,
                          biasInitializer: biasInit)
    }

    @differentiable
    public func callAsFunction(_ input: Tensorf) -> Tensorf {
        var x = input.padded(forSizes: [(0, 0), (3, 3), (3, 3), (0, 0)], mode: .reflect)
        x = x.sequenced(through: conv1, norm1)
        x = relu(x)
        x = x.sequenced(through: conv2, norm2)
        x = relu(x)

        x = x.sequenced(through: conv3, norm3)
        x = relu(x)

        x = resblocks(x)

        x = x.sequenced(through: upConv1, upNorm1)
        x = relu(x)
        x = x.sequenced(through: upConv2, upNorm2)
        x = relu(x)

        x = lastConv(x)
        x = tanh(x)

        return x
    }
}
