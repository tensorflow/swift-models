import TensorFlow

public protocol AutoLayer {
    associatedtype InstanceType: Layer
    associatedtype InputShape
    associatedtype OutputShape

    func buildModelWithOutputShape(inputShape: InputShape) -> (InstanceType, OutputShape)
}

extension AutoLayer {
    public func buildModel(inputShape: InputShape) -> InstanceType {
        return self.buildModelWithOutputShape(inputShape: inputShape).0
    }
}
