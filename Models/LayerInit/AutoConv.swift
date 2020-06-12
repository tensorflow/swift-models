import TensorFlow

public struct AutoConv2D<Scalar>: AutoLayer where Scalar: TensorFlowFloatingPoint {
    let filterShape: (Int, Int)
    let outputChannels: Int
    let strides: (Int, Int)
    let padding: Padding
    let dilations: (Int, Int)
    let activation: Conv2D<Scalar>.Activation
    let useBias: Bool
    let filterInitializer: ParameterInitializer<Scalar>
    let biasInitializer: ParameterInitializer<Scalar>

    public typealias InstanceType = Conv2D<Scalar>
    public typealias InputShape = (Int, Int, Int)
    public typealias OutputShape = (Int, Int, Int)

    public init(
        filterShape: (Int, Int),
        outputChannels: Int,
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid,
        dilations: (Int, Int) = (1, 1),
        activation: @escaping Conv2D<Scalar>.Activation = identity,
        useBias: Bool = true,
        filterInitializer: @escaping ParameterInitializer<Scalar> = glorotUniform(),
        biasInitializer: @escaping ParameterInitializer<Scalar> = zeros()
    ) {
        self.filterShape = filterShape
        self.outputChannels = outputChannels
        self.strides = strides
        self.padding = padding
        self.dilations = dilations
        self.activation = activation
        self.useBias = useBias
        self.filterInitializer = filterInitializer
        self.biasInitializer = biasInitializer
    }

    public func buildModelWithOutputShape<Prefix>(inputShape: (Int, Int, Int), keyPathSoFar: KeyPath<Prefix, InstanceType>, keyDict: inout [AnyAutoLayerKey: Any]) -> (InstanceType, (Int, Int, Int)) {
        let outputShape: (Int, Int, Int)
        if (padding == .valid) {
            outputShape = (
                Int(ceil(Float(inputShape.0 - filterShape.0 + 1) / Float(strides.0))),
                Int(ceil(Float(inputShape.1 - filterShape.1 + 1) / Float(strides.1))),
                outputChannels
            )
        } else {
            outputShape = (
                Int(ceil(Float(inputShape.0) / Float(strides.0))),
                Int(ceil(Float(inputShape.1) / Float(strides.1))),
                outputChannels
            )
        }

        return (Conv2D<Scalar>(
            filterShape: (filterShape.0, filterShape.1, inputShape.2, outputChannels),
            strides: strides, padding: padding, dilations: dilations,
            activation: activation, useBias: useBias,
            filterInitializer: filterInitializer, biasInitializer: biasInitializer
        ), outputShape)
    }
}
