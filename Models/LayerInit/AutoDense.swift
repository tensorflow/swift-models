import TensorFlow

public struct AutoDense<Scalar>: AutoLayer where Scalar: TensorFlowFloatingPoint {
    let outputSize: Int;
    let activation: Dense<Scalar>.Activation

    public typealias InstanceType = Dense<Scalar>
    public typealias InputShape = Int
    public typealias OutputShape = Int

    public init(outputSize: Int, activation: @escaping Dense<Scalar>.Activation = identity) {
        self.outputSize = outputSize
        self.activation = activation
    }

    public func buildModelWithOutputShape<Prefix>(inputShape: Int, keyPathSoFar: KeyPath<Prefix, InstanceType>, keyDict: inout [AnyAutoLayerKey: Any]) -> (InstanceType, Int) {
        return (Dense<Scalar>(inputSize: inputShape, outputSize: self.outputSize, activation: self.activation), self.outputSize)
    }
}
