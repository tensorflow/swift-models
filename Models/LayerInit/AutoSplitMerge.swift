import TensorFlow

// Workaround https://bugs.swift.org/browse/TF-1122
public final class SplitMergeFunctionWrapper<Output1: Differentiable, Output2: Differentiable, CommonOutput: Differentiable> {
  public typealias F = @differentiable (Output1, Output2) -> CommonOutput
  public var f: F
  public init(_ f: @escaping F) { self.f = f }
}

public struct SplitMergeInstance<Layer1: Layer, Layer2: Layer, CommonOutput: Differentiable>: Layer
where Layer1.Input == Layer2.Input, Layer1.TangentVector.VectorSpaceScalar == Layer2.TangentVector.VectorSpaceScalar {
    public var layer1: Layer1
    public var layer2: Layer2
    @noDerivative let mergeFn: SplitMergeFunctionWrapper<Layer1.Output, Layer2.Output, CommonOutput>

    public init(layer1: Layer1, layer2: Layer2, mergeFn: SplitMergeFunctionWrapper<Layer1.Output, Layer2.Output, CommonOutput>) {
        self.layer1 = layer1
        self.layer2 = layer2
        self.mergeFn = mergeFn
    }

    @differentiable
    public func callAsFunction(_ input: Layer1.Input) -> CommonOutput {
        let layer1Out = layer1(input)
        let layer2Out = layer2(input)
        return mergeFn.f(layer1Out, layer2Out)
    }
}

public struct AutoSplitMerge<Layer1: AutoLayer, Layer2: AutoLayer, CommonOutput: Differentiable, OutputShape>: AutoLayer
where Layer1.InputShape == Layer2.InputShape, Layer1.InstanceType.Input == Layer2.InstanceType.Input, Layer1.InstanceType.TangentVector.VectorSpaceScalar == Layer2.InstanceType.TangentVector.VectorSpaceScalar {
    let layer1: Layer1
    let layer2: Layer2

    let mergeOutputShape: (Layer1.OutputShape, Layer2.OutputShape) -> OutputShape
    let mergeFn: SplitMergeFunctionWrapper<Layer1.InstanceType.Output, Layer2.InstanceType.Output, CommonOutput>

    public typealias InstanceType = SplitMergeInstance<Layer1.InstanceType, Layer2.InstanceType, CommonOutput>
    public typealias InputShape = Layer1.InputShape
    public typealias OutputShape = OutputShape

    public init(
        layer1: Layer1, layer2: Layer2,
        mergeOutputShape: @escaping (Layer1.OutputShape, Layer2.OutputShape) -> OutputShape,
        mergeFn: @escaping @differentiable (Layer1.InstanceType.Output, Layer2.InstanceType.Output) -> CommonOutput
    ) {
        self.layer1 = layer1
        self.layer2 = layer2
        self.mergeOutputShape = mergeOutputShape
        self.mergeFn = SplitMergeFunctionWrapper(mergeFn)
    }

    public func buildModelWithOutputShape<Prefix>(inputShape: Layer1.InputShape, keyPathSoFar: KeyPath<Prefix, InstanceType>, keyDict: inout [AnyAutoLayerKey: Any]) -> (InstanceType, OutputShape) {
        let (layer1Built, layer1OutputShape) = layer1.buildModelWithOutputShape(inputShape: inputShape, keyPathSoFar: keyPathSoFar.appending(path: \InstanceType.layer1), keyDict: &keyDict)
        let (layer2Built, layer2OutputShape) = layer2.buildModelWithOutputShape(inputShape: inputShape, keyPathSoFar: keyPathSoFar.appending(path: \InstanceType.layer2), keyDict: &keyDict)
        return (SplitMergeInstance(layer1: layer1Built, layer2: layer2Built, mergeFn: self.mergeFn), self.mergeOutputShape(layer1OutputShape, layer2OutputShape))
    }
}
