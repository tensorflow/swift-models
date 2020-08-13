import TensorFlow

/// A specification of the shape of the input to a traced graph.
public class InputTracingLayer: TracingLayer<()> {
    let _outputShape: [Int]

    public init(shape: [Int]) {
        self._outputShape = shape
    }

    public override var outputShape: [Int] {
        return _outputShape
    }

    override func makeClassicLayer() -> DynamicLayerStore {
        return DynamicLayerStore(Dense<Float>(inputSize: 1, outputSize: 1)) // TODO
    }

    public override var dependencies: [AnyTracingLayer] {
        return []
    }

    override func buildLayerApplication(dependencyIndices: [Int])
        -> @differentiable ([Tensor<Float>], DynamicLayerStore) -> Tensor<Float> {
        let inputIndex = dependencyIndices[0]
        return { (dependencySource: [Tensor<Float>], classicLayer: DynamicLayerStore) in
            return dependencySource[inputIndex]
        }
    }
}

/// A specification for a layer that passes a single dependency's result through a classic layer
public class TracingLayerWrapper<L: Layer>: TracingLayer<L>
where L.Input == Tensor<Float>, L.Output == Tensor<Float>, L.TangentVector.VectorSpaceScalar == Float {
    let layer: L
    let dependency: AnyTracingLayer
    let _outputShape: [Int]

    public init(dependency: AnyTracingLayer, layer: L, outputShape: [Int]) {
        self.dependency = dependency
        self.layer = layer
        self._outputShape = outputShape
    }

    public override var outputShape: [Int] {
        return _outputShape
    }

    override func makeClassicLayer() -> DynamicLayerStore {
        return DynamicLayerStore(layer)
    }

    public override var dependencies: [AnyTracingLayer] {
        return [dependency]
    }

    override func buildLayerApplication(dependencyIndices: [Int])
        -> @differentiable ([Tensor<Float>], DynamicLayerStore) -> Tensor<Float> {
        let prevIndex = dependencyIndices[0]
        return { (outputs: [Tensor<Float>], selfLayer: DynamicLayerStore) -> Tensor<Float> in
            return selfLayer.callDynamically(outputs[prevIndex], layerType: L.self)
        }
    }
}

/// A specification for a layer that shares its weights with another tracing layer
public class SharedWeightsWrapper<L: Layer>: TracingLayer<L>
where L.Input == Tensor<Float>, L.Output == Tensor<Float>, L.TangentVector.VectorSpaceScalar == Float {
    let _sharingWeightsWith: TracingLayer<L>
    let dependency: AnyTracingLayer

    public init(dependency: AnyTracingLayer, sharingWeightsWith: TracingLayer<L>) {
        self.dependency = dependency
        self._sharingWeightsWith = sharingWeightsWith
    }

    public override var outputShape: [Int] {
        return _sharingWeightsWith.outputShape
    }

    override var sharingWeightsWith: AnyTracingLayer? {
        return _sharingWeightsWith
    }

    override func makeClassicLayer() -> DynamicLayerStore {
        return DynamicLayerStore(Dense<Float>(inputSize: 1, outputSize: 1)) // TODO
    }

    public override var dependencies: [AnyTracingLayer] {
        return [dependency, _sharingWeightsWith] // TODO: to force expansion of shared weights first
    }

    override func buildLayerApplication(dependencyIndices: [Int])
        -> @differentiable ([Tensor<Float>], DynamicLayerStore) -> Tensor<Float> {
        let prevIndex = dependencyIndices[0]
        return { (outputs: [Tensor<Float>], selfLayer: DynamicLayerStore) -> Tensor<Float> in
            return selfLayer.callDynamically(outputs[prevIndex], layerType: L.self)
        }
    }
}

extension TracingLayer where L: Layer, L.Input == Tensor<Float>, L.Output == Tensor<Float>, L.TangentVector.VectorSpaceScalar == Float {
    public func sharingWeights(with: TracingLayer<L>) -> TracingLayer<L> {
        // TODO(shadaj): should not assume single dependency
        return SharedWeightsWrapper(dependency: self.dependencies[0], sharingWeightsWith: with)
    }
}

/// A tracing layer which combines the results of two dependencies with a custom function
public class MergeTracingLayer: TracingLayer<()> {
    let mergeFn: @differentiable (Tensor<Float>, Tensor<Float>) -> Tensor<Float>
    let dependency1: AnyTracingLayer
    let dependency2: AnyTracingLayer
    
    let _outputShape: [Int]

    public init(
        dependency1: AnyTracingLayer, dependency2: AnyTracingLayer,
        mergeFn: @escaping @differentiable (Tensor<Float>, Tensor<Float>) -> Tensor<Float>,
        outputShape: [Int]
    ) {
        self.dependency1 = dependency1
        self.dependency2 = dependency2
        self.mergeFn = mergeFn
        self._outputShape = outputShape
    }

    public override var outputShape: [Int] {
        return _outputShape
    }

    override func makeClassicLayer() -> DynamicLayerStore {
        return DynamicLayerStore(Dense<Float>(inputSize: 1, outputSize: 1)) // TODO
    }

    public override var dependencies: [AnyTracingLayer] {
        return [dependency1, dependency2]
    }

    override func buildLayerApplication(dependencyIndices: [Int])
        -> @differentiable ([Tensor<Float>], DynamicLayerStore) -> Tensor<Float> {
        let prev1Index = dependencyIndices[0]
        let prev2Index = dependencyIndices[1]
        return { outputs, _ in
            return self.mergeFn(outputs[prev1Index], outputs[prev2Index])
        }
    }
}
