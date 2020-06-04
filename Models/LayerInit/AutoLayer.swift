import TensorFlow

public protocol AutoLayer {
    associatedtype InstanceType: Layer
    func buildModel(inputShape: Int) -> (InstanceType, Int)
}

public struct AutoSequencedDefinition<Layer1: AutoLayer, Layer2: AutoLayer>: AutoLayer
where
  Layer1.InstanceType.Output == Layer2.InstanceType.Input,
  Layer1.InstanceType.TangentVector.VectorSpaceScalar == Layer2.InstanceType.TangentVector.VectorSpaceScalar {
    let first: Layer1
    let second: Layer2

    public typealias InstanceType = Sequential<Layer1.InstanceType, Layer2.InstanceType>

    public init(first: Layer1, second: Layer2) {
        self.first = first
        self.second = second
    }

    public func buildModel(inputShape: Int) -> (InstanceType, Int) {
        let (firstInstance, firstOutputShape) = first.buildModel(inputShape: inputShape)
        let (secondInstance, secondOutputShape) = second.buildModel(inputShape: firstOutputShape)
        return (Sequential(firstInstance, secondInstance), secondOutputShape)
    }
}

extension AutoSequencedDefinition {
    public func then<T: AutoLayer>(_ other: T) -> AutoSequencedDefinition<AutoSequencedDefinition<Layer1, Layer2>, T> {
        return AutoSequencedDefinition<AutoSequencedDefinition<Layer1, Layer2>, T>(first: self, second: other)
    }
}

public struct AutoDenseDefinition<Scalar>: AutoLayer where Scalar: TensorFlowFloatingPoint {
    let outputSize: Int;
    let activation: Dense<Scalar>.Activation

    public typealias InstanceType = Dense<Scalar>

    public init(outputSize: Int, activation: @escaping Dense<Scalar>.Activation = identity) {
        self.outputSize = outputSize
        self.activation = activation
    }

    public func buildModel(inputShape: Int) -> (InstanceType, Int) {
        return (Dense<Scalar>(inputSize: inputShape, outputSize: self.outputSize, activation: self.activation), self.outputSize)
    }
}

extension AutoDenseDefinition {
    public func then<T: AutoLayer>(_ other: T) -> AutoSequencedDefinition<AutoDenseDefinition<Scalar>, T> {
        return AutoSequencedDefinition<AutoDenseDefinition<Scalar>, T>(first: self, second: other)
    }
}
