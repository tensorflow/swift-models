import TensorFlow
import Foundation

/// A model that applies style.
public struct TransformerNet: Layer {
    public typealias Input = Tensor<Float>
    public typealias Output = Tensor<Float>

    // Convolution & instance normalization layers.
    public var conv1 = ConvLayer(inChannels: 3, outChannels: 32, kernelSize: 9, stride: 1)
    public var in1 = InstanceNorm2D<Float>(featureCount: 32)
    public var conv2 = ConvLayer(inChannels: 32, outChannels: 64, kernelSize: 3, stride: 2)
    public var in2 = InstanceNorm2D<Float>(featureCount: 64)
    public var conv3 = ConvLayer(inChannels: 64, outChannels: 128, kernelSize: 3, stride: 2)
    public var in3 = InstanceNorm2D<Float>(featureCount: 128)

    // Residual block layers.
    public var res1 = ResidualBlock(channels: 128)
    public var res2 = ResidualBlock(channels: 128)
    public var res3 = ResidualBlock(channels: 128)
    public var res4 = ResidualBlock(channels: 128)
    public var res5 = ResidualBlock(channels: 128)

    // Upsampling & instance normalization layers.
    public var deconv1 = UpsampleConvLayer(
        inChannels: 128, outChannels: 64,
        kernelSize: 3, stride: 1, scaleFactor: 2.0)
    public var in4 = InstanceNorm2D<Float>(featureCount: 64)
    public var deconv2 = UpsampleConvLayer(
        inChannels: 64, outChannels: 32,
        kernelSize: 3, stride: 1, scaleFactor: 2.0)
    public var in5 = InstanceNorm2D<Float>(featureCount: 32)
    public var deconv3 = UpsampleConvLayer(
        inChannels: 32, outChannels: 3,
        kernelSize: 9, stride: 1)

    // ReLU activation layer.
    public var relu = ReLU<Float>()

    /// Creates style transformer model.
    public init() {}

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer. Expected layout is BxHxWxC.
    /// - Returns: The output.
    @differentiable
    public func callAsFunction(_ input: Input) -> Output {
        let convolved1 = input.sequenced(through: conv1, in1, relu)
        let convolved2 = convolved1.sequenced(through: conv2, in2, relu)
        let convolved3 = convolved2.sequenced(through: conv3, in3, relu)
        let residual = convolved3.sequenced(through: res1, res2, res3, res4, res5)
        let upscaled1 = residual.sequenced(through: deconv1, in4, relu)
        let upscaled2 = upscaled1.sequenced(through: deconv2, in5)
        let upscaled3 = deconv3(upscaled2)
        return upscaled3
    }
}

/// Helper layer: convolution with padding.
public struct ConvLayer: Layer {
    public typealias Input = Tensor<Float>
    public typealias Output = Tensor<Float>

    /// Padding layer.
    public var reflectionPad: ReflectionPad2D<Float>
    /// Convolution layer.
    public var conv2d: Conv2D<Float>

    /// Creates 2D convolution with padding layer.
    ///
    /// - Parameters:
    ///   - inChannels: Number of input channels in convolution kernel.
    ///   - outChannels: Number of output channels in convolution kernel.
    ///   - kernelSize: Convolution kernel size (both width and height).
    ///   - stride: Stride size (both width and height).
    public init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int) {
        reflectionPad = ReflectionPad2D<Float>(padding: Int(kernelSize / 2))
        conv2d = Conv2D<Float>(
            filterShape: (kernelSize, kernelSize, inChannels, outChannels),
            strides: (stride, stride)
        )
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func callAsFunction(_ input: Input) -> Output {
        return input.sequenced(through: reflectionPad, conv2d)
    }
}

/// Helper layer: residual block.
public struct ResidualBlock: Layer {
    public typealias Input = Tensor<Float>
    public typealias Output = Tensor<Float>

    /// Convolution & instance normalization layers.
    public var conv1: ConvLayer
    public var in1: InstanceNorm2D<Float>
    public var conv2: ConvLayer
    public var in2: InstanceNorm2D<Float>

    /// Activation layer.
    public var relu = ReLU<Float>()

    /// Creates 2D residual block layer.
    ///
    /// - Parameter channels: Number of input channels in convolution kernel.
    public init(channels: Int) {
        conv1 = ConvLayer(inChannels: channels, outChannels: channels, kernelSize: 3, stride: 1)
        in1 = InstanceNorm2D<Float>(featureCount: channels)
        conv2 = ConvLayer(inChannels: channels, outChannels: channels, kernelSize: 3, stride: 1)
        in2 = InstanceNorm2D<Float>(featureCount: channels)
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func callAsFunction(_ input: Input) -> Output {
        return input + input.sequenced(through:
            conv1, in1, relu,
            conv2, in2
        )
    }
}

/// Helper layer for upsampling.
///
/// Upsamples the input and then does a convolution.
/// Reference: http://distill.pub/2016/deconv-checkerboard/
public struct UpsampleConvLayer: Layer {
    public typealias Input = Tensor<Float>
    public typealias Output = Tensor<Float>

    /// Scale factor.
    @noDerivative public let scaleFactor: Float
    /// Padding layer.
    public var reflectionPad: ReflectionPad2D<Float>
    /// Convolution layer.
    public var conv2d: Conv2D<Float>

    /// Creates 2D upsampling layer.
    ///
    /// - Parameters:
    ///   - inChannels: Number of input channels in convolution kernel.
    ///   - outChannels: Number of output channels in convolution kernel.
    ///   - kernelSize: Convolution kernel size (both width and height).
    ///   - stride: Stride size (both width and height).
    ///   - scaleFactor: Scale factor.
    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int,
        scaleFactor: Float = 1.0
    ) {
        self.scaleFactor = scaleFactor
        reflectionPad = ReflectionPad2D<Float>(padding: Int(kernelSize / 2))
        conv2d = Conv2D<Float>(
            filterShape: (kernelSize, kernelSize, inChannels, outChannels),
            strides: (stride, stride)
        )
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func callAsFunction(_ input: Input) -> Output {
        let resizedInput = resizeNearestNeighbor(input, scaleFactor: scaleFactor)
        return resizedInput.sequenced(through: reflectionPad, conv2d)
    }
}
