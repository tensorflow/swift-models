import TensorFlow

/// A layer "blueprint", which defines elements that can be constructed into a Layer instance for training
public protocol AutoLayer {
    /// The type of the layer instance that can be built with this prototype
    associatedtype InstanceType: Layer

    /// The specific tuple of `Int`s that define the input shape of the layer
    associatedtype InputShape

    /// The specific tuple of `Int`s that define the output shape of the layer
    associatedtype OutputShape

    /// Initializes a new instance of the layer defined by this blueprint.
    /// Parameters:
    ///     - inputShape: the shape of a single input instance (no batch) to this layer
    ///     - keyPathSoFar: a `KeyPath` that tracks the path from the root layer to the current layer instance
    ///     - keyDict: a dictionary tracking the mapping from `AutoLayerKey`s to the key path to the layer instance
    /// Returns:
    ///     - $0: the instance of the layer with the given input shape
    ///     - $1: the output shape of the layer instance computed based on the input shape
    func buildModelWithOutputShape<Prefix>(inputShape: InputShape, keyPathSoFar: KeyPath<Prefix, InstanceType>, keyDict: inout [AnyAutoLayerKey: Any]) -> (InstanceType, OutputShape)
}

extension AutoLayer {
    /// Builds an instance of the model with the given input shape
    public func buildModel(inputShape: InputShape) -> BuiltAutoLayer<InstanceType> {
        var keyDict: [AnyAutoLayerKey: Any] = [:]
        let (layerInstance, _) = self.buildModelWithOutputShape(inputShape: inputShape, keyPathSoFar: \InstanceType.self, keyDict: &keyDict)
        return BuiltAutoLayer(layer: layerInstance, keyMapping: keyDict)
    }
}

/// A layer instance containing a model built with the AutoLayer API. Offers keyed access to layers with `AutoLayerKey`.
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

    /// Grab a specific layer by the given `AutoLayerKey`.
    public subscript<T>(index: AutoLayerKey<T>) -> T {
        return self.layer[keyPath: self.keyMapping[index] as! KeyPath<InstanceType, T>]
    }
}
