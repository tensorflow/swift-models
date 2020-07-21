import TensorFlow

/// A specification of the shape of the input to a traced graph.
public class InputTracingLayer: TracingLayer {
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

    public override var dependencies: [TracingLayer] {
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
public class TracingLayerWrapper<L: Layer>: TracingLayer
where L.Input == Tensor<Float>, L.Output == Tensor<Float>, L.TangentVector.VectorSpaceScalar == Float {
    let layer: L
    let dependency: TracingLayer
    let _outputShape: [Int]

    public init(dependency: TracingLayer, layer: L, outputShape: [Int]) {
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

    public override var dependencies: [TracingLayer] {
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

/// A tracing layer which combines the results of two dependencies with a custom function
public class MergeTracingLayer: TracingLayer {
    let mergeFn: @differentiable (Tensor<Float>, Tensor<Float>) -> Tensor<Float>
    let dependency1: TracingLayer
    let dependency2: TracingLayer
    
    let _outputShape: [Int]

    public init(
        dependency1: TracingLayer, dependency2: TracingLayer,
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

    public override var dependencies: [TracingLayer] {
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
