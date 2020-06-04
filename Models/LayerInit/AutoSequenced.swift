import TensorFlow

public struct AutoSequencedDefinition<Layer1: AutoLayer, Layer2: AutoLayer>: AutoLayer
where
  Layer1.OutputShape == Layer2.InputShape,
  Layer1.InstanceType.Output == Layer2.InstanceType.Input,
  Layer1.InstanceType.TangentVector.VectorSpaceScalar == Layer2.InstanceType.TangentVector.VectorSpaceScalar {
    let first: Layer1
    let second: Layer2

    public typealias InstanceType = Sequential<Layer1.InstanceType, Layer2.InstanceType>

    public init(first: Layer1, second: Layer2) {
        self.first = first
        self.second = second
    }

    public func buildModelWithOutputShape(inputShape: Layer1.InputShape) -> (InstanceType, Layer2.OutputShape) {
        let (firstInstance, firstOutputShape) = first.buildModelWithOutputShape(inputShape: inputShape)
        let (secondInstance, secondOutputShape) = second.buildModelWithOutputShape(inputShape: firstOutputShape)
        return (Sequential(firstInstance, secondInstance), secondOutputShape)
    }
}

extension AutoLayer {
    public func then<T: AutoLayer>(_ other: T) -> AutoSequencedDefinition<Self, T> {
        return AutoSequencedDefinition<Self, T>(first: self, second: other)
    }
}
