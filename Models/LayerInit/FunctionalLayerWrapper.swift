import TensorFlow

public class InputFunctionalLayer: FunctionalLayer {
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

    public override func getDependencies() -> [FunctionalLayer] {
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

public class FunctionalLayerWrapper<L: Layer>: FunctionalLayer
where L.Input == Tensor<Float>, L.Output == Tensor<Float>, L.TangentVector.VectorSpaceScalar == Float {
    let layer: L
    let parent: FunctionalLayer
    let _outputShape: [Int]

    public init(parent: FunctionalLayer, layer: L, outputShape: [Int]) {
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

    public override func getDependencies() -> [FunctionalLayer] {
        return [parent]
    }

    public override func buildLayerApplication(dependencyIndices: [Int])
        -> @differentiable ([Tensor<Float>], DynamicLayerStore) -> Tensor<Float> {
        let prevIndex = dependencyIndices[0]
        return { (outputs: [Tensor<Float>], selfLayer: DynamicLayerStore) in
            return selfLayer.callWithLayer(
                outputs[prevIndex],
                { (layer: L, input: Tensor<Float>) in layer.callAsFunction(input) }
            )
        }
    }
}

public class MergeLayerWrapper: FunctionalLayer {
    let mergeFn: @differentiable (Tensor<Float>, Tensor<Float>) -> Tensor<Float>
    let parent1: FunctionalLayer
    let parent2: FunctionalLayer
    
    let _outputShape: [Int]

    public init(
        parent1: FunctionalLayer, parent2: FunctionalLayer,
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

    public override func getDependencies() -> [FunctionalLayer] {
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
