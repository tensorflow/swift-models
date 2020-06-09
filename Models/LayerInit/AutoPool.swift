import TensorFlow

public struct AutoAvgPool2D<Scalar>: AutoLayer where Scalar: TensorFlowFloatingPoint {
    let poolSize: (Int, Int)
    let strides: (Int, Int)
    let padding: Padding

    public typealias InstanceType = AvgPool2D<Scalar>
    public typealias InputShape = (Int, Int, Int)
    public typealias OutputShape = (Int, Int, Int)

    public init(
        poolSize: (Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid
    ) {
        self.poolSize = poolSize
        self.strides = strides
        self.padding = padding
    }

    public func buildModelWithOutputShape(inputShape: (Int, Int, Int)) -> (InstanceType, (Int, Int, Int)) {
        let outputShape: (Int, Int, Int)
        if (padding == .valid) {
            outputShape = (
                Int(ceil(Float(inputShape.0 - poolSize.0 + 1) / Float(strides.0))),
                Int(ceil(Float(inputShape.1 - poolSize.1 + 1) / Float(strides.1))),
                inputShape.2
            )
        } else {
            outputShape = (
                Int(ceil(Float(inputShape.0) / Float(strides.0))),
                Int(ceil(Float(inputShape.1) / Float(strides.1))),
                inputShape.2
            )
        }

        return (AvgPool2D<Scalar>(
            poolSize: poolSize,
            strides: strides,
            padding: padding
        ), outputShape)
    }
}

public struct AutoMaxPool2D<Scalar>: AutoLayer where Scalar: TensorFlowFloatingPoint {
    let poolSize: (Int, Int)
    let strides: (Int, Int)
    let padding: Padding

    public typealias InstanceType = MaxPool2D<Scalar>
    public typealias InputShape = (Int, Int, Int)
    public typealias OutputShape = (Int, Int, Int)

    public init(
        poolSize: (Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid
    ) {
        self.poolSize = poolSize
        self.strides = strides
        self.padding = padding
    }

    public func buildModelWithOutputShape(inputShape: (Int, Int, Int)) -> (InstanceType, (Int, Int, Int)) {
        let outputShape: (Int, Int, Int)
        if (padding == .valid) {
            outputShape = (
                Int(ceil(Float(inputShape.0 - poolSize.0 + 1) / Float(strides.0))),
                Int(ceil(Float(inputShape.1 - poolSize.1 + 1) / Float(strides.1))),
                inputShape.2
            )
        } else {
            outputShape = (
                Int(ceil(Float(inputShape.0) / Float(strides.0))),
                Int(ceil(Float(inputShape.1) / Float(strides.1))),
                inputShape.2
            )
        }

        return (MaxPool2D<Scalar>(
            poolSize: poolSize,
            strides: strides,
            padding: padding
        ), outputShape)
    }
}
