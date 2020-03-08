import TensorFlow
import Foundation

public struct Identity: ParameterlessLayer {
    @differentiable
    public func callAsFunction(_ input: Tensorf) -> Tensorf {
        input
    }
}

public struct LeakyRELU: ParameterlessLayer {
    @differentiable
    public func callAsFunction(_ input: Tensorf) -> Tensorf {
        return leakyRelu(input)
    }
}

public struct Tanh: ParameterlessLayer {
    @differentiable
    public func callAsFunction(_ input: Tensorf) -> Tensorf {
        return tanh(input)
    }
}

public struct RELU: ParameterlessLayer {
    @differentiable
    public func callAsFunction(_ input: Tensorf) -> Tensorf {
        return relu(input)
    }
}

struct ELU: ParameterlessLayer {
    @differentiable
    func callAsFunction(_ input: Tensorf) -> Tensorf {
        return elu(input)
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

/// A 2-D layer applying padding with zeros over a mini-batch.
public struct ZeroPad2D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
    /// The padding values along the spatial dimensions.
    @noDerivative public let padding: ((Int, Int), (Int, Int))

    /// Creates a zero-padding 2D Layer.
    ///
    /// - Parameter padding: A tuple of 2 tuples of two integers describing how many elements to
    ///   be padded at the beginning and end of each padding dimensions.
    public init(padding: ((Int, Int), (Int, Int))) {
        self.padding = padding
    }

    /// Creates a zero-padding 2D Layer.
    ///
    /// - Parameter padding: Integer that describes how many elements to be padded
    ///   at the beginning and end of each padding dimensions.
    public init(padding: Int) {
        self.padding = ((padding, padding), (padding, padding))
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer. Expected layout is BxHxWxC.
    /// - Returns: The output.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        // Padding applied to height and width dimensions only.
        return input.padded(forSizes: [
            (0, 0),
            padding.0,
            padding.1,
            (0, 0)
        ], mode: .constant(0))
    }
}
/// A 2-D layer applying padding with reflection over a mini-batch.
public struct ReflectionPad2D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
    /// The padding values along the spatial dimensions.
    @noDerivative public let padding: ((Int, Int), (Int, Int))

    /// Creates a reflect-padding 2D Layer.
    ///
    /// - Parameter padding: A tuple of 2 tuples of two integers describing how many elements to
    ///   be padded at the beginning and end of each padding dimensions.
    public init(padding: ((Int, Int), (Int, Int))) {
        self.padding = padding
    }

    /// Creates a reflect-padding 2D Layer.
    ///
    /// - Parameter padding: Integer that describes how many elements to be padded
    ///   at the beginning and end of each padding dimensions.
    public init(padding: Int) {
        self.padding = ((padding, padding), (padding, padding))
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer. Expected layout is BxHxWxC.
    /// - Returns: The output.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        // Padding applied to height and width dimensions only.
        return input.padded(forSizes: [
            (0, 0),
            padding.0,
            padding.1,
            (0, 0)
        ], mode: .reflect)
    }
}

public struct ConvLayer: Layer {
    public typealias Input = Tensorf
    public typealias Output = Tensorf

    /// Padding layer.
    public var pad: ZeroPad2D<Float>
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
        self.pad = ZeroPad2D<Float>(padding: padding ?? Int(kernelSize / 2))

        self.conv2d = Conv2D<Float>(filterShape: (kernelSize, kernelSize,
                                                  inChannels, outChannels),
                                    strides: (stride, stride),
                                    filterInitializer: { Tensorf(randomNormal: $0, standardDeviation: Tensorf(0.02)) })
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
    public var downRELU = LeakyRELU()
    public var downConv: Conv2D<Float>
    public var upConv: TransposedConv2D<Float>
    public var upRELU = RELU()
    public var upNorm: BatchNorm<Float>
    
    public init(inChannels: Int,
                innerChannels: Int,
                outChannels: Int) {
        self.downConv = .init(filterShape: (4, 4, inChannels, innerChannels),
                              strides: (2, 2),
                              padding: .same,
                              filterInitializer: { Tensorf(randomNormal: $0, standardDeviation: Tensorf(0.02)) })
        self.upNorm = .init(featureCount: outChannels)
        
        self.upConv = .init(filterShape: (4, 4, innerChannels, outChannels),
                            strides: (2, 2),
                            padding: .same,
                            filterInitializer: { Tensorf(randomNormal: $0, standardDeviation: Tensorf(0.02)) })
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensorf) -> Tensorf {
        var x = self.downRELU(input)
        x = self.downConv(x)
        x = self.upRELU(x)
        x = self.upConv(x)
        x = self.upNorm(x)

        return input.concatenated(with: x, alongAxis: 3)
    }
}


public struct UNetSkipConnection<NT: Layer>: Layer where NT.TangentVector.VectorSpaceScalar == Float, NT.Input == Tensorf, NT.Output == Tensorf {
    public var downRELU = LeakyRELU()
    public var downConv: Conv2D<Float>
    public var downNorm: BatchNorm<Float>
    public var upConv: TransposedConv2D<Float>
    public var upRELU = RELU()
    public var upNorm: BatchNorm<Float>
    public var dropOut = Dropout<Float>(probability: 0.5)
    @noDerivative public var useDropOut: Bool
    
    public var submodule: NT
    
    public init(inChannels: Int,
                innerChannels: Int,
                outChannels: Int,
                submodule: NT,
                useDropOut: Bool = false) {
        self.downConv = .init(filterShape: (4, 4, inChannels, innerChannels),
                              strides: (2, 2),
                              padding: .same,
                              filterInitializer: { Tensorf(randomNormal: $0, standardDeviation: Tensorf(0.02)) })
        self.downNorm = .init(featureCount: innerChannels)
        self.upNorm = .init(featureCount: outChannels)
        
        self.upConv = .init(filterShape: (4, 4, outChannels, innerChannels * 2),
                            strides: (2, 2),
                            padding: .same,
                            filterInitializer: { Tensorf(randomNormal: $0, standardDeviation: Tensorf(0.02)) })
    
        self.submodule = submodule
        
        self.useDropOut = useDropOut
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensorf) -> Tensorf {
        var x = self.downRELU(input)
        x = self.downConv(x)
        x = self.downNorm(x)
        x = self.submodule(x)
        x = self.upRELU(x)
        x = self.upConv(x)
        x = self.upNorm(x)
        
        if self.useDropOut {
            x = self.dropOut(x)
        }
        
        return input.concatenated(with: x, alongAxis: 3)
    }
}

public struct UNetSkipConnectionOutermost<NT: Layer>: Layer where NT.TangentVector.VectorSpaceScalar == Float, NT.Input == Tensorf, NT.Output == Tensorf {
    public var downConv: Conv2D<Float>
    public var upConv: TransposedConv2D<Float>
    public var upRELU = RELU()
    
    public var submodule: NT
    
    public init(inChannels: Int,
                innerChannels: Int,
                outChannels: Int,
                submodule: NT) {
        self.downConv = .init(filterShape: (4, 4, inChannels, innerChannels),
                              strides: (2, 2),
                              padding: .same,
                              filterInitializer: { Tensorf(randomNormal: $0, standardDeviation: Tensorf(0.02)) })
        self.upConv = .init(filterShape: (4, 4, outChannels, innerChannels * 2),
                            strides: (2, 2),
                            padding: .same,
                            activation: tanh,
                            filterInitializer: { Tensorf(randomNormal: $0, standardDeviation: Tensorf(0.02)) })
    
        self.submodule = submodule
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensorf) -> Tensorf {
        var x = self.downConv(input)
        x = self.submodule(x)
        x = self.upRELU(x)
        x = self.upConv(x)

        return x
    }
}
