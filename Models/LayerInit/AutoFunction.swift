import TensorFlow

/// A layer that applies a user-defined function to the input data
public struct AutoFunction<Input: Differentiable, Output: Differentiable, InputShape, OutputShape>: AutoLayer {
    let fnShape: (InputShape) -> OutputShape
    let fn: @differentiable (Input) -> Output

    public typealias InstanceType = Function<Input, Output>
    public typealias InputShape = InputShape
    public typealias OutputShape = OutputShape

    /**
     Constructs a function layer instance.

     Parameters:
        - fnShape: a function that computes the output shape of the function given the input shape
        - fn: a function that computes the output data of the function given the input data
     */
    public init(fnShape: @escaping (InputShape) -> OutputShape, fn: @escaping @differentiable (Input) -> Output) {
        self.fnShape = fnShape
        self.fn = fn
    }

    public func buildModelWithOutputShape<Prefix>(inputShape: InputShape, keyPathSoFar: KeyPath<Prefix, InstanceType>, keyDict: inout [AnyAutoLayerKey: Any]) -> (InstanceType, OutputShape) {
        return (Function(fn), fnShape(inputShape))
    }
}
