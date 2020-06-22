import TensorFlow

public struct AutoSequenced<Layer1: AutoLayer, Layer2: AutoLayer>: AutoLayer
where
  Layer1.OutputShape == Layer2.InputShape,
  Layer1.InstanceType.Output == Layer2.InstanceType.Input,
  Layer1.InstanceType.TangentVector.VectorSpaceScalar == Layer2.InstanceType.TangentVector.VectorSpaceScalar {
    let first: Layer1
    let second: Layer2

    public typealias InstanceType = Sequential<Layer1.InstanceType, Layer2.InstanceType>
    public typealias InputShape = Layer1.InputShape
    public typealias OutputShape = Layer2.OutputShape

    public init(first: Layer1, second: Layer2) {
        self.first = first
        self.second = second
    }

    public func buildModelWithOutputShape<Prefix>(inputShape: Layer1.InputShape, keyPathSoFar: KeyPath<Prefix, InstanceType>, keyDict: inout [AnyAutoLayerKey: Any]) -> (InstanceType, Layer2.OutputShape) {
        let (firstInstance, firstOutputShape) = first.buildModelWithOutputShape(inputShape: inputShape, keyPathSoFar: keyPathSoFar.appending(path: \InstanceType.layer1), keyDict: &keyDict)
        let (secondInstance, secondOutputShape) = second.buildModelWithOutputShape(inputShape: firstOutputShape, keyPathSoFar: keyPathSoFar.appending(path: \InstanceType.layer2), keyDict: &keyDict)
        return (Sequential(firstInstance, secondInstance), secondOutputShape)
    }
}

extension AutoLayer {
    /// Composes layers by sequencing the current layer to be followed by the passed layer
    public func then<T: AutoLayer>(_ other: T) -> AutoSequenced<Self, T> {
        return AutoSequenced<Self, T>(first: self, second: other)
    }
}

public struct AutoSequencedManyInstance<LayerType: Layer>: Layer
where LayerType.Input == LayerType.Output {
    public var layers: [LayerType]

    @differentiable
    public func callAsFunction(_ input: LayerType.Input) -> LayerType.Output {
        return layers.differentiableReduce(input) { $1($0) }
    }
}

/// A layer blueprint consisting of many instances of the same type of layer sequenced together
/// The sequenced layer type must have the same input shape and output shape dimensions, since otherwise there is a shape mismatch.
public struct AutoSequencedMany<LayerType: AutoLayer>: AutoLayer
where
  LayerType.OutputShape == LayerType.InputShape,
  LayerType.InstanceType.Input == LayerType.InstanceType.Output {
    let layers: [LayerType]

    public typealias InstanceType = AutoSequencedManyInstance<LayerType.InstanceType>

    public init(layers: [LayerType]) {
        self.layers = layers
    }

    public func buildModelWithOutputShape<Prefix>(inputShape: LayerType.InputShape, keyPathSoFar: KeyPath<Prefix, InstanceType>, keyDict: inout [AnyAutoLayerKey: Any]) -> (InstanceType, LayerType.OutputShape) {
        var lastOutputShape = inputShape
        let builtInstances = self.layers.enumerated().map({ (idx, autoLayer) -> LayerType.InstanceType in
            let (instance, outputShape) = autoLayer.buildModelWithOutputShape(inputShape: lastOutputShape, keyPathSoFar: keyPathSoFar.appending(path: \InstanceType.layers[idx]), keyDict: &keyDict)
            lastOutputShape = outputShape
            return instance
        })

        return (AutoSequencedManyInstance(layers: builtInstances), lastOutputShape)
    }
}
