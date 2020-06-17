import TensorFlow

public struct AutoFlatten<Scalar>: AutoLayer where Scalar: TensorFlowFloatingPoint {
    public typealias InstanceType = Flatten<Scalar>
    public typealias InputShape = Any
    public typealias OutputShape = Int

    public init() {}

    public func buildModelWithOutputShape<Prefix>(inputShape: Any, keyPathSoFar: KeyPath<Prefix, InstanceType>, keyDict: inout [AnyAutoLayerKey: Any]) -> (InstanceType, Int) {
        return (Flatten<Scalar>(), intTupleToArray(tuple: inputShape).reduce(1, { $0 * $1 }))
    }
}
