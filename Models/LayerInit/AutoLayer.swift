import TensorFlow

public protocol AutoLayer {
    associatedtype InstanceType: Layer
    associatedtype InputShape
    associatedtype OutputShape

    func buildModelWithOutputShape<Prefix>(inputShape: InputShape, keyPathSoFar: KeyPath<Prefix, InstanceType>, keyDict: inout [AnyAutoLayerKey: Any]) -> (InstanceType, OutputShape)
}

extension AutoLayer {
    public func buildModel(inputShape: InputShape) -> BuiltAutoLayer<InstanceType> {
        var keyDict: [AnyAutoLayerKey: Any] = [:]
        let (layerInstance, _) = self.buildModelWithOutputShape(inputShape: inputShape, keyPathSoFar: \InstanceType.self, keyDict: &keyDict)
        return BuiltAutoLayer(layer: layerInstance, keyMapping: keyDict)
    }

    public func buildModelWithKeys(inputShape: InputShape) -> (InstanceType, [AnyAutoLayerKey: Any]) {
        var keyDict: [AnyAutoLayerKey: Any] = [:]
        let (layerInstance, _) = self.buildModelWithOutputShape(inputShape: inputShape, keyPathSoFar: \InstanceType.self, keyDict: &keyDict)
        return (layerInstance, keyDict)
    }
}

public struct BuiltAutoLayer<InstanceType: Layer>: Layer {
    public var layer: InstanceType
    @noDerivative let keyMapping: [AnyAutoLayerKey: Any]

    public init(layer: InstanceType, keyMapping: [AnyAutoLayerKey: Any]) {
        self.layer = layer
        self.keyMapping = keyMapping
    }

    @differentiable
    public func callAsFunction(_ input: InstanceType.Input) -> InstanceType.Output {
        return layer(input)
    }

    public subscript<T>(index: AutoLayerKey<T>) -> T {
        return self.layer[keyPath: self.keyMapping[index] as! KeyPath<InstanceType, T>]
    }
}
