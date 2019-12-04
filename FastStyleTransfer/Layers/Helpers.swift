import TensorFlow

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

/// A layer applying `relu` activation function.
public struct ReLU<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return relu(input)
    }
}
