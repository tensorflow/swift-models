import TensorFlow
import Foundation

/// Model that applies style
public struct TransformerNet: Layer {
    public typealias Input = Tensor<Float>
    public typealias Output = Tensor<Float>

    // conv layers
    public var conv1 = ConvLayer(in_channels: 3, out_channels: 32, kernel_size: 9, stride: 1)
    public var in1 = InstanceNorm2d<Float>(featureCount: 32)
    public var conv2 = ConvLayer(in_channels: 32, out_channels: 64, kernel_size: 3, stride: 2)
    public var in2 = InstanceNorm2d<Float>(featureCount: 64)
    public var conv3 = ConvLayer(in_channels: 64, out_channels: 128, kernel_size: 3, stride: 2)
    public var in3 = InstanceNorm2d<Float>(featureCount: 128)

    // residual layers
    public var res1 = ResidualBlock(channels: 128)
    public var res2 = ResidualBlock(channels: 128)
    public var res3 = ResidualBlock(channels: 128)
    public var res4 = ResidualBlock(channels: 128)
    public var res5 = ResidualBlock(channels: 128)

    // upsampling layers
    public var deconv1 = UpsampleConvLayer(in_channels: 128, out_channels: 64, kernel_size: 3, stride: 1, upsample: 2.0)
    public var in4 = InstanceNorm2d<Float>(featureCount: 64)
    public var deconv2 = UpsampleConvLayer(in_channels: 64, out_channels: 32, kernel_size: 3, stride: 1, upsample: 2.0)
    public var in5 = InstanceNorm2d<Float>(featureCount: 32)
    public var deconv3 = UpsampleConvLayer(in_channels: 32, out_channels: 3, kernel_size: 9, stride: 1)

    // activation
    public var relu = ReLU<Float>()

    public init() {}

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

/// Helper convolution layer with padding
public struct ConvLayer: Layer {
    public typealias Input = Tensor<Float>
    public typealias Output = Tensor<Float>

    public var reflection_pad: ReflectionPad2d<Float>
    public var conv2d: Conv2D<Float>

    public init(in_channels: Int, out_channels: Int, kernel_size: Int, stride: Int) {
        reflection_pad = ReflectionPad2d<Float>(padding: Int(kernel_size / 2))
        conv2d = Conv2D<Float>(filterShape: (kernel_size, kernel_size, in_channels, out_channels), strides: (stride, stride))
    }

    @differentiable
    public func callAsFunction(_ input: Input) -> Output {
        return input.sequenced(through: reflection_pad, conv2d)
    }

}

/// Residual block
public struct ResidualBlock: Layer {
    public typealias Input = Tensor<Float>
    public typealias Output = Tensor<Float>

    public var conv1: ConvLayer
    public var in1: InstanceNorm2d<Float>
    public var conv2: ConvLayer
    public var in2: InstanceNorm2d<Float>
    public var relu = ReLU<Float>()

    public init(channels: Int) {
        conv1 = ConvLayer(in_channels: channels, out_channels: channels, kernel_size: 3, stride: 1)
        in1 = InstanceNorm2d<Float>(featureCount: channels)
        conv2 = ConvLayer(in_channels: channels, out_channels: channels, kernel_size: 3, stride: 1)
        in2 = InstanceNorm2d<Float>(featureCount: channels)
    }

    @differentiable
    public func callAsFunction(_ input: Input) -> Output {
        return input + input.sequenced(
                through:
                conv1, in1, relu,
                conv2, in2
        )
    }
}

/// Upscaling layer
public struct UpsampleConvLayer: Layer {
    public typealias Input = Tensor<Float>
    public typealias Output = Tensor<Float>

    @noDerivative public let upsample: Float
    public var reflection_pad: ReflectionPad2d<Float>
    public var conv2d: Conv2D<Float>

    public init(in_channels: Int, out_channels: Int, kernel_size: Int, stride: Int, upsample: Float = 1.0) {
        self.upsample = upsample
        reflection_pad = ReflectionPad2d<Float>(padding: Int(kernel_size / 2))
        conv2d = Conv2D<Float>(filterShape: (kernel_size, kernel_size, in_channels, out_channels), strides: (stride, stride))
    }

    @differentiable
    public func callAsFunction(_ input: Input) -> Output {
        let resizedInput = resizeNearestNeighbor(input, scale_factor: upsample)
        return resizedInput.sequenced(through: reflection_pad, conv2d)
    }
}

