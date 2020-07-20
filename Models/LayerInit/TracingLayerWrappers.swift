import TensorFlow

/// A tracing layer which represents the input to a model
public class InputTracingLayer: TracingLayer {
    let shape: [Int]

    public init(shape: [Int]) {
        self.shape = shape
    }

    public override func outputShape() -> [Int] {
        return shape
    }

    public override func getLayer() -> DynamicLayerStore {
        return DynamicLayerStore(Dense<Float>(inputSize: 1, outputSize: 1)) // TODO
    }

    public override func getDependencies() -> [TracingLayer] {
        return []
    }

    public override func buildLayerApplication(dependencyIndices: [Int])
        -> @differentiable ([Tensor<Float>], DynamicLayerStore) -> Tensor<Float> {
        let inputIndex = dependencyIndices[0]
        return { (outputs: [Tensor<Float>], selfLayer: DynamicLayerStore) in
            return outputs[inputIndex]
        }
    }
}

/// A tracing layer which passes the parent's result through a classic layer
public class TracingLayerWrapper<L: Layer>: TracingLayer
where L.Input == Tensor<Float>, L.Output == Tensor<Float>, L.TangentVector.VectorSpaceScalar == Float {
    let layer: L
    let parent: TracingLayer
    let _outputShape: [Int]

    public init(parent: TracingLayer, layer: L, outputShape: [Int]) {
        self.parent = parent
        self.layer = layer
        self._outputShape = outputShape
    }

    public override func outputShape() -> [Int] {
        return _outputShape
    }

    public override func getLayer() -> DynamicLayerStore {
        return DynamicLayerStore(layer)
    }

    public override func getDependencies() -> [TracingLayer] {
        return [parent]
    }

    public override func buildLayerApplication(dependencyIndices: [Int])
        -> @differentiable ([Tensor<Float>], DynamicLayerStore) -> Tensor<Float> {
        let prevIndex = dependencyIndices[0]
        return { (outputs: [Tensor<Float>], selfLayer: DynamicLayerStore) -> Tensor<Float> in
            let layerSpecializer: L? = nil // Guide type parameter inference to pick up the layer type
            return selfLayer.callDynamically(outputs[prevIndex], layerSpecializer: layerSpecializer)
        }
    }
}

/// A tracing layer which combines the results of two parent layers with a custom function
public class MergeTracingLayer: TracingLayer {
    let mergeFn: @differentiable (Tensor<Float>, Tensor<Float>) -> Tensor<Float>
    let parent1: TracingLayer
    let parent2: TracingLayer
    
    let _outputShape: [Int]

    public init(
        parent1: TracingLayer, parent2: TracingLayer,
        mergeFn: @escaping @differentiable (Tensor<Float>, Tensor<Float>) -> Tensor<Float>,
        outputShape: [Int]
    ) {
        self.parent1 = parent1
        self.parent2 = parent2
        self.mergeFn = mergeFn
        self._outputShape = outputShape
    }

    public override func outputShape() -> [Int] {
        return _outputShape
    }

    public override func getLayer() -> DynamicLayerStore {
        return DynamicLayerStore(Dense<Float>(inputSize: 1, outputSize: 1)) // TODO
    }

    public override func getDependencies() -> [TracingLayer] {
        return [parent1, parent2]
    }

    public override func buildLayerApplication(dependencyIndices: [Int])
        -> @differentiable ([Tensor<Float>], DynamicLayerStore) -> Tensor<Float> {
        let prev1Index = dependencyIndices[0]
        let prev2Index = dependencyIndices[1]
        return { (outputs: [Tensor<Float>], selfLayer: DynamicLayerStore) in
            return self.mergeFn(outputs[prev1Index], outputs[prev2Index])
        }
    }
}
