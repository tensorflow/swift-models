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
import Foundation

public struct Identity: ParameterlessLayer {
    public typealias TangentVector = EmptyTangentVector

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        input
    }
}

/// 2-D layer applying instance normalization over a mini-batch of inputs.
///
/// Reference: [Instance Normalization](https://arxiv.org/abs/1607.08022)
public struct InstanceNorm2D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// Learnable parameter scale for affine transformation.
    public var scale: Tensor<Scalar>
    /// Learnable parameter offset for affine transformation.
    public var offset: Tensor<Scalar>
    /// Small value added in denominator for numerical stability.
    @noDerivative public var epsilon: Tensor<Scalar>

    /// Creates a instance normalization 2D Layer.
    ///
    /// - Parameters:
    ///   - featureCount: Size of the channel axis in the expected input.
    ///   - epsilon: Small scalar added for numerical stability.
    public init(featureCount: Int, epsilon: Tensor<Scalar> = Tensor(1e-5)) {
        self.epsilon = epsilon
        scale = Tensor<Scalar>(ones: [featureCount])
        offset = Tensor<Scalar>(zeros: [featureCount])
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer. Expected input layout is BxHxWxC.
    /// - Returns: The output.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        // Calculate mean & variance along H,W axes.
        let mean = input.mean(alongAxes: [1, 2])
        let variance = input.variance(alongAxes: [1, 2])
        let norm = (input - mean) * rsqrt(variance + epsilon)
        return norm * scale + offset
    }
}

public struct ConvLayer: Layer {
    public typealias Input = Tensor<Float>
    public typealias Output = Tensor<Float>

    /// Padding layer.
    public var pad: ZeroPadding2D<Float>
    /// Convolution layer.
    public var conv2d: Conv2D<Float>

    /// Creates 2D convolution with padding layer.
    ///
    /// - Parameters:
    ///   - inChannels: Number of input channels in convolution kernel.
    ///   - outChannels: Number of output channels in convolution kernel.
    ///   - kernelSize: Convolution kernel size (both width and height).
    ///   - stride: Stride size (both width and height).
    public init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int, padding: Int? = nil) {
        let _padding =  padding ?? Int(kernelSize / 2)
        pad = ZeroPadding2D(padding: ((_padding, _padding), (_padding, _padding)))
    
        conv2d = Conv2D(filterShape: (kernelSize, kernelSize, inChannels, outChannels),
                        strides: (stride, stride),
                        filterInitializer: { Tensor<Float>(randomNormal: $0, standardDeviation: Tensor<Float>(0.02)) })
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func callAsFunction(_ input: Input) -> Output {
        return input.sequenced(through: pad, conv2d)
    }
}

public struct UNetSkipConnectionInnermost: Layer {
    public var downConv: Conv2D<Float>
    public var upConv: TransposedConv2D<Float>
    public var upNorm: BatchNorm<Float>
    
    public init(inChannels: Int,
                innerChannels: Int,
                outChannels: Int) {
        self.downConv = Conv2D(filterShape: (4, 4, inChannels, innerChannels),
                               strides: (2, 2),
                               padding: .same,
                               filterInitializer: { Tensor<Float>(randomNormal: $0,
                                                            standardDeviation: Tensor<Float>(0.02)) })
        self.upNorm = BatchNorm(featureCount: outChannels)
        
        self.upConv = TransposedConv2D(filterShape: (4, 4, innerChannels, outChannels),
                                       strides: (2, 2),
                                       padding: .same,
                                       filterInitializer: { Tensor<Float>(randomNormal: $0,
                                                                    standardDeviation: Tensor<Float>(0.02)) })
    }
    
    public init(downConv: Conv2D<Float>, upConv: TransposedConv2D<Float>, upNorm: BatchNorm<Float>) {
        self.downConv = downConv
        self.upConv = upConv
        self.upNorm = upNorm
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = leakyRelu(input)
        x = self.downConv(x)
        x = relu(x)
        x = x.sequenced(through: self.upConv, self.upNorm)
        
        return input.concatenated(with: x, alongAxis: 3)
    }
}

extension Conv2D: Equatable {
    public static func == (left: Conv2D, right: Conv2D) -> Bool {
        return left.filter == right.filter &&
               left.bias == right.bias &&
               left.strides == right.strides &&
               left.padding == right.padding &&
               left.dilations == right.dilations
    }
}

extension TransposedConv2D: Equatable {
    public static func == (left: TransposedConv2D, right: TransposedConv2D) -> Bool {
        return left.filter == right.filter &&
               left.bias == right.bias &&
                left.strides == right.strides &&
                left.padding == right.padding
    }
}

extension BatchNorm: Equatable {
    public static func == (left: BatchNorm, right: BatchNorm) -> Bool {
         return left.axis == right.axis &&
                left.momentum == right.momentum &&
                left.offset == right.offset &&
                left.scale == right.scale &&
                left.epsilon == right.epsilon  &&
                left.runningMean.value == right.runningMean.value &&
                left.runningVariance.value == right.runningVariance.value
    }
}

extension UNetSkipConnectionInnermost: Equatable {
    public static func == (left: UNetSkipConnectionInnermost, right: UNetSkipConnectionInnermost) -> Bool {
         return left.downConv == right.downConv &&
                left.upConv == right.upConv &&
                left.upNorm == right.upNorm
    }
}

extension Dropout: Equatable {
    public static func == (left: Dropout, right: Dropout) -> Bool {
        return left.probability == right.probability
    }
}

public struct UNetSkipConnection<Sublayer: Layer>: Layer where Sublayer.TangentVector.VectorSpaceScalar == Float, Sublayer.Input == Tensor<Float>, Sublayer.Output == Tensor<Float> {
    public var downConv: Conv2D<Float>
    public var downNorm: BatchNorm<Float>
    public var upConv: TransposedConv2D<Float>
    public var upNorm: BatchNorm<Float>
    public var dropOut = Dropout<Float>(probability: 0.5)
    @noDerivative public var useDropOut: Bool
    
    public var submodule: Sublayer
    
    public init(inChannels: Int,
                innerChannels: Int,
                outChannels: Int,
                submodule: Sublayer,
                useDropOut: Bool = false) {
        self.downConv = Conv2D(filterShape: (4, 4, inChannels, innerChannels),
                               strides: (2, 2),
                               padding: .same,
                               filterInitializer: { Tensor<Float>(randomNormal: $0, standardDeviation: Tensor<Float>(0.02)) })
        self.downNorm = BatchNorm(featureCount: innerChannels)
        self.upNorm = BatchNorm(featureCount: outChannels)
        
        self.upConv = TransposedConv2D(filterShape: (4, 4, outChannels, innerChannels * 2),
                                       strides: (2, 2),
                                       padding: .same,
                                       filterInitializer: { Tensor<Float>(randomNormal: $0,
                                                                    standardDeviation: Tensor<Float>(0.02)) })
    
        self.submodule = submodule
        
        self.useDropOut = useDropOut
    }
    
    public init(downConv: Conv2D<Float>, downNorm: BatchNorm<Float>, upConv: TransposedConv2D<Float>, upNorm: BatchNorm<Float>, dropOut: Dropout<Float>, submodule: Sublayer, useDropOut: Bool = false) {
        self.downConv = downConv
        self.downNorm = downNorm
        self.upConv = upConv
        self.upNorm = upNorm
        self.dropOut = dropOut
        self.submodule = submodule
        self.useDropOut = useDropOut
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = leakyRelu(input)
        x = x.sequenced(through: self.downConv, self.downNorm, self.submodule)
        x = relu(x)
        x = x.sequenced(through: self.upConv, self.upNorm)
        
        if self.useDropOut {
            x = self.dropOut(x)
        }
        
        return input.concatenated(with: x, alongAxis: 3)
    }
}

    
    
    
/*
extension Array where Element: Comparable {
    func sort() -> [Element] {
            return sorted(by: {$0 < $1})
    }
}
*/
    
extension UNetSkipConnection: Equatable {
    public static func == (left: UNetSkipConnection, right: UNetSkipConnection) -> Bool {
        return left.downConv == right.downConv &&
               left.downNorm == right.downNorm &&
               left.upConv == right.upConv &&
               left.upNorm == right.upNorm &&
               left.dropOut == right.dropOut &&
               left.useDropOut == right.useDropOut
    }
}
    


public struct UNetSkipConnectionOutermost<Sublayer: Layer>: Layer where Sublayer.TangentVector.VectorSpaceScalar == Float, Sublayer.Input == Tensor<Float>, Sublayer.Output == Tensor<Float> {
    public var downConv: Conv2D<Float>
    public var upConv: TransposedConv2D<Float>
    
    public var submodule: Sublayer
    
    public init(inChannels: Int,
                innerChannels: Int,
                outChannels: Int,
                submodule: Sublayer) {
        self.downConv = Conv2D(filterShape: (4, 4, inChannels, innerChannels),
                               strides: (2, 2),
                               padding: .same,
                               filterInitializer: { Tensor<Float>(randomNormal: $0,
                                                            standardDeviation: Tensor<Float>(0.02)) })
        self.upConv = TransposedConv2D(filterShape: (4, 4, outChannels, innerChannels * 2),
                                       strides: (2, 2),
                                       padding: .same,
                                       activation: tanh,
                                       filterInitializer: { Tensor<Float>(randomNormal: $0,
                                                                    standardDeviation: Tensor<Float>(0.02)) })
    
        self.submodule = submodule
    }
    
    public init(downConv: Conv2D<Float>, upConv: TransposedConv2D<Float>, submodule: Sublayer) {
        self.downConv = downConv
//        self.upConv = upConv
        // TODO: need to persist the activation function.  Until then, manually set it to tanh.
        self.upConv = TransposedConv2D(filter: upConv.filter, bias: upConv.bias, activation: tanh, strides: upConv.strides, padding: upConv.padding)
        self.submodule = submodule
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input.sequenced(through: self.downConv, self.submodule)
        x = relu(x)
        x = self.upConv(x)
        return x
    }
}

extension UNetSkipConnectionOutermost: Equatable {
    public static func == (left: UNetSkipConnectionOutermost, right: UNetSkipConnectionOutermost) -> Bool {
        return left.downConv == right.downConv &&
               left.upConv == right.upConv
    }
}
