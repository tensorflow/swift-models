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
// V2 paper
// "Bag of Tricks for Image Classification with Convolutional Neural Networks"
// Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, Mu Li
// https://arxiv.org/abs/1812.01187

// A convolution and batchnorm layer
public struct ConvBNV2: Layer {
    public var conv: Conv2D<Float>
    public var norm: BatchNorm<Float>
    @noDerivative public let useRelu: Bool

    public init(
        inFilters: Int,
        outFilters: Int,
        kernelSize: Int = 1,
        stride: Int = 1,
        padding: Padding = .same,
        useRelu: Bool = true
    ) {
        //Should use no bias
        self.conv = Conv2D(
            filterShape: (kernelSize, kernelSize, inFilters, outFilters), 
            strides: (stride, stride), 
            padding: padding)
        self.norm = BatchNorm(featureCount: outFilters, momentum: 0.9, epsilon: 1e-5)
        self.useRelu = useRelu
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let convResult = input.sequenced(through: conv, norm)
        return useRelu ? relu(convResult) : convResult
    }
}

// The shortcut in a Residual Block
// Workaround optionals not being differentiable, can be simplified when it's the case
// Resnet-D trick: use average pooling instead of stride 2 conv for the shortcut
public struct Shortcut: Layer {
    public var projection: ConvBNV2
    public var avgPool: AvgPool2D<Float>
    @noDerivative public let needsProjection: Bool
    @noDerivative public let needsPool: Bool
    
    public init(inFilters: Int, outFilters: Int, stride: Int) {
        avgPool = AvgPool2D<Float>(poolSize: (2, 2), strides: (stride, stride))
        needsPool = (stride != 1)
        needsProjection = (inFilters != outFilters)
        projection = ConvBNV2(
            inFilters:  needsProjection ? inFilters  : 1, 
            outFilters: needsProjection ? outFilters : 1
        )
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var res = input
        if needsProjection { res = projection(res) }
        if needsPool       { res = avgPool(res)}
        return res
    }
}

// Residual block for a ResNet V2
// Resnet-B trick: stride on the inside conv
public struct ResidualBlockV2: Layer {
    public var shortcut: Shortcut
    public var convs: [ConvBNV2]

    public init(inFilters: Int, outFilters: Int, stride: Int, expansion: Int){
        if expansion == 1 {
            convs = [
                ConvBNV2(inFilters: inFilters,  outFilters: outFilters, kernelSize: 3, stride: stride),
                ConvBNV2(inFilters: outFilters, outFilters: outFilters, kernelSize: 3, useRelu: false)
            ]
        } else {
            convs = [
                ConvBNV2(inFilters: inFilters,    outFilters: outFilters/4),
                ConvBNV2(inFilters: outFilters/4, outFilters: outFilters/4, kernelSize: 3, stride: stride),
                ConvBNV2(inFilters: outFilters/4, outFilters: outFilters, useRelu: false)
            ]
        }
        shortcut = Shortcut(inFilters: inFilters, outFilters: outFilters, stride: stride)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let convResult = convs.differentiableReduce(input) { $1($0) }
        return relu(convResult + shortcut(input))
    }
}

/// An implementation of the ResNet v2 architectures, at various depths.
public struct ResNetV2: Layer {
    public var inputStem: [ConvBNV2]
    public var maxPool: MaxPool2D<Float>
    public var residualBlocks: [ResidualBlockV2] = []
    public var avgPool = GlobalAvgPool2D<Float>()
    public var flatten = Flatten<Float>()
    public var classifier: Dense<Float>

    /// Initializes a new ResNet v2 network model.
    ///
    /// - Parameters:
    ///   - classCount: The number of classes the network will be or has been trained to identify.
    ///   - depth: A specific depth for the network, chosen from the enumerated values in 
    ///     ResNet.Depth.
    ///   - inputChannels: The number of channels of the input
    ///   - stemFilters: The number of filters in the first three convolutions.
    ///         Resnet-A trick uses 64-64-64, research at fastai suggests 32-32-64 is better
    public init(
        classCount: Int, 
        depth: Depth, 
        inputChannels: Int = 3, 
        stemFilters: [Int] = [32, 32, 64]
    ) {
        let filters = [inputChannels] + stemFilters
        inputStem = Array(0..<3).map { i in
            ConvBNV2(inFilters: filters[i], outFilters: filters[i+1], kernelSize: 3, stride: i==0 ? 2 : 1)
        }
        maxPool = MaxPool2D(poolSize: (3, 3), strides: (2, 2), padding: .same)
        let sizes = [64 / depth.expansion, 64, 128, 256, 512]
        for (iBlock, nBlocks) in depth.layerBlockSizes.enumerated() {
            let (nIn, nOut) = (sizes[iBlock] * depth.expansion, sizes[iBlock+1] * depth.expansion)
            for j in 0..<nBlocks {
                residualBlocks.append(ResidualBlockV2(
                    inFilters: j==0 ? nIn : nOut,  
                    outFilters: nOut, 
                    stride: (iBlock != 0) && (j == 0) ? 2 : 1, 
                    expansion: depth.expansion
                ))
            }
        }
        classifier = Dense(inputSize: 512 * depth.expansion, outputSize: classCount)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let inputLayer = maxPool(inputStem.differentiableReduce(input) { $1($0) })
        let blocksReduced = residualBlocks.differentiableReduce(inputLayer) { $1($0) }
        return blocksReduced.sequenced(through: avgPool, flatten, classifier)
    }
}

extension ResNetV2 {
    public enum Depth {
        case resNet18
        case resNet34
        case resNet50
        case resNet101
        case resNet152

        var expansion: Int {
            switch self {
            case .resNet18, .resNet34: return 1
            default: return 4
            }
        }

        var layerBlockSizes: [Int] {
            switch self {
            case .resNet18:  return [2, 2, 2,  2]
            case .resNet34:  return [3, 4, 6,  3]
            case .resNet50:  return [3, 4, 6,  3]
            case .resNet101: return [3, 4, 23, 3]
            case .resNet152: return [3, 8, 36, 3]
            }
        }
    }
}
