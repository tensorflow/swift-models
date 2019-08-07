import TensorFlow

/// Layer for padding with reflection over mini-batch of images
/// Expected input layout is BxHxWxC
public struct ReflectionPad2d<Scalar: TensorFlowFloatingPoint>: Layer {
    @noDerivative public let padding: ((Int, Int), (Int, Int))

    public init(padding: ((Int, Int), (Int, Int))) {
        self.padding = padding
    }

    public init(padding: Int) {
        self.padding = ((padding, padding), (padding, padding))
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input.paddedWithReflection(forSizes: [
            (0, 0),
            padding.0,
            padding.1,
            (0, 0)
        ])
    }
}


/// Layer applying relu activation function
public struct ReLU<Scalar: TensorFlowFloatingPoint>: Layer {
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return relu(input)
    }
}