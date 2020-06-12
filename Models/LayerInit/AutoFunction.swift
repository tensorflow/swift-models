import TensorFlow

public struct AutoFunction<Input: Differentiable, Output: Differentiable, InputShape, OutputShape>: AutoLayer {
    let fnShape: (InputShape) -> OutputShape
    let fn: @differentiable (Input) -> Output

    public typealias InstanceType = Function<Input, Output>
    public typealias InputShape = InputShape
    public typealias OutputShape = OutputShape

    public init(fnShape: @escaping (InputShape) -> OutputShape, fn: @escaping @differentiable (Input) -> Output) {
        self.fnShape = fnShape
        self.fn = fn
    }

    public func buildModelWithOutputShape<Prefix>(inputShape: InputShape, keyPathSoFar: KeyPath<Prefix, InstanceType>, keyDict: inout [AnyAutoLayerKey: Any]) -> (InstanceType, OutputShape) {
        return (Function(fn), fnShape(inputShape))
    }
}
