import TensorFlow

public protocol AutoLayer {
    associatedtype InstanceType: Layer
    associatedtype InputShape
    associatedtype OutputShape

    func buildModelWithOutputShape<Prefix>(inputShape: InputShape, keyPathSoFar: KeyPath<Prefix, InstanceType>, keyDict: inout [AnyAutoLayerKey: Any]) -> (InstanceType, OutputShape)
}

extension AutoLayer {
    public func buildModel(inputShape: InputShape) -> InstanceType {
        var keyDict: [AnyAutoLayerKey: Any] = [:]
        let (layerInstance, _) = self.buildModelWithOutputShape(inputShape: inputShape, keyPathSoFar: \InstanceType.self, keyDict: &keyDict)
        return layerInstance
    }

    public func buildModelWithKeys(inputShape: InputShape) -> (InstanceType, [AnyAutoLayerKey: Any]) {
        var keyDict: [AnyAutoLayerKey: Any] = [:]
        let (layerInstance, _) = self.buildModelWithOutputShape(inputShape: inputShape, keyPathSoFar: \InstanceType.self, keyDict: &keyDict)
        return (layerInstance, keyDict)
    }
}
