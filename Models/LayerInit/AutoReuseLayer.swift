import TensorFlow

public struct AutoReuseLayerInstance<OuterLayer: Layer, MiddleLayer: Layer>: Layer
where
  OuterLayer.Output == MiddleLayer.Input, OuterLayer.Input == MiddleLayer.Output,
  OuterLayer.TangentVector.VectorSpaceScalar == MiddleLayer.TangentVector.VectorSpaceScalar {
    public var outerLayer: OuterLayer
    public var middleLayer: MiddleLayer

    @differentiable
    public func callAsFunction(_ input: OuterLayer.Input) -> OuterLayer.Output {
        return outerLayer(middleLayer(outerLayer(input)))
    }
}

/// A layer blueprint for models that need to reuse the same layer in different places.
/// Sequences a layer, then another layer, and then the first layer again.
public struct AutoReuseLayer<OuterLayer: AutoLayer, MiddleLayer: AutoLayer>: AutoLayer
where
  OuterLayer.OutputShape == MiddleLayer.InputShape,
  MiddleLayer.OutputShape == OuterLayer.InputShape,
  OuterLayer.InstanceType.Output == MiddleLayer.InstanceType.Input,
  MiddleLayer.InstanceType.Output == OuterLayer.InstanceType.Input,
  OuterLayer.InstanceType.TangentVector.VectorSpaceScalar == MiddleLayer.InstanceType.TangentVector.VectorSpaceScalar {
    let outer: OuterLayer
    let middle: MiddleLayer

    public typealias InstanceType = AutoReuseLayerInstance<OuterLayer.InstanceType, MiddleLayer.InstanceType>

    /**
     Constructs the layer reuse blueprint.

     Parameters:
        - outer: the layer to be reused, once at the beginning and once at the end
        - middle: the layer to be used once, in between the two outer layers
     */
    public init(outer: OuterLayer, middle: MiddleLayer) {
        self.outer = outer
        self.middle = middle
    }

    public func buildModelWithOutputShape<Prefix>(inputShape: OuterLayer.InputShape, keyPathSoFar: KeyPath<Prefix, InstanceType>, keyDict: inout [AnyAutoLayerKey: Any]) -> (InstanceType, OuterLayer.OutputShape) {
        let (outerLayer, firstOuterOutputShape) = outer.buildModelWithOutputShape(inputShape: inputShape, keyPathSoFar: keyPathSoFar.appending(path: \InstanceType.outerLayer), keyDict: &keyDict)
        let (middleLayer, middleOutputShape) = middle.buildModelWithOutputShape(inputShape: firstOuterOutputShape, keyPathSoFar: keyPathSoFar.appending(path: \InstanceType.middleLayer), keyDict: &keyDict)

        var tempDictionary: [AnyAutoLayerKey: Any] = [:]

        let shapeMismatchError: String = "Cannot reuse outer layer because original input size \(inputShape) does not match output size of middle layer \(middleOutputShape)"
        precondition(intTupleToArray(tuple: inputShape) == intTupleToArray(tuple: middleOutputShape), shapeMismatchError)
        let (_, finalOutputShape) = outer.buildModelWithOutputShape(inputShape: middleOutputShape, keyPathSoFar: keyPathSoFar.appending(path: \InstanceType.outerLayer), keyDict: &tempDictionary)

        return (AutoReuseLayerInstance(outerLayer: outerLayer, middleLayer: middleLayer), finalOutputShape)
    }
}
