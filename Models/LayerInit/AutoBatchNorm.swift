import TensorFlow

public struct AutoBatchNorm<Shape, Scalar>: AutoLayer where Scalar: TensorFlowFloatingPoint {
    let axis: Int
    let momentum: Scalar
    let epsilon: Scalar

    public typealias InstanceType = BatchNorm<Scalar>
    public typealias InputShape = Shape
    public typealias OutputShape = Shape

    public init(
        axis: Int = -1,
        momentum: Scalar = 0.99,
        epsilon: Scalar = 0.001
    ) {
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
    }

    public func buildModelWithOutputShape<Prefix>(inputShape: Shape, keyPathSoFar: KeyPath<Prefix, InstanceType>, keyDict: inout [AnyAutoLayerKey: Any]) -> (InstanceType, Shape) {
        let inputShapeArray: [Int] = intTupleToArray(tuple: inputShape)

        let featureCount = inputShapeArray[(inputShapeArray.count + axis) % inputShapeArray.count]
        return (BatchNorm<Scalar>(featureCount: featureCount, axis: axis, momentum: momentum, epsilon: epsilon), inputShape)
    }
}
