import TensorFlow

/// Layer that applies instance normalization over a mini-batch of images
/// Expected input layout is BxHxWxC
/// Reference: [Instance Normalization](https://arxiv.org/abs/1607.08022)
public struct InstanceNorm2d<Scalar: TensorFlowFloatingPoint>: Layer {
    public var scale: Tensor<Scalar>
    public var offset: Tensor<Scalar>
    @noDerivative public var epsilon: Tensor<Scalar>

    public init(featureCount: Int, epsilon: Tensor<Scalar> = Tensor(1e-5)) {
        self.epsilon = epsilon
        scale = Tensor<Scalar>(ones: [featureCount])
        offset = Tensor<Scalar>(zeros: [featureCount])
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let mean = input.mean(alongAxes: [1, 2])
        let variance = input.variance(alongAxes: [1, 2])
        let norm = (input - mean) * rsqrt(variance + epsilon)
        return norm * scale + offset
    }
}
