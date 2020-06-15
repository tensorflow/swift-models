import TensorFlow

public class AnyAutoLayerKey: Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(ObjectIdentifier(self))
    }

    public static func == (lhs: AnyAutoLayerKey, rhs: AnyAutoLayerKey) -> Bool {
        return lhs === rhs
    }
}

public class AutoLayerKey<T: Layer>: AnyAutoLayerKey {
    public override init() {}
}

public struct KeyedAutoLayer<Underlying: AutoLayer>: AutoLayer {
    let underlying: Underlying
    let key: AutoLayerKey<InstanceType>

    public typealias InstanceType = Underlying.InstanceType
    public typealias InputShape = Underlying.InputShape
    public typealias OutputShape = Underlying.OutputShape

    public init(_ underlying: Underlying, key: AutoLayerKey<InstanceType>) {
        self.underlying = underlying
        self.key = key
    }

    public func buildModelWithOutputShape<Prefix>(inputShape: InputShape, keyPathSoFar: KeyPath<Prefix, InstanceType>, keyDict: inout [AnyAutoLayerKey: Any]) -> (InstanceType, OutputShape) {
        let (layer, outputShape) = underlying.buildModelWithOutputShape(inputShape: inputShape, keyPathSoFar: keyPathSoFar, keyDict: &keyDict)
        keyDict[self.key] = keyPathSoFar
        return (layer, outputShape)
    }
}
