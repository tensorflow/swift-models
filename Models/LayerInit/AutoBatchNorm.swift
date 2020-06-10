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

    public func buildModelWithOutputShape(inputShape: Shape) -> (InstanceType, Shape) {
        let inputShapeArray: [Int]
        if let inputShapeTuple = inputShape as? (Int, Int, Int) {
            inputShapeArray = [inputShapeTuple.0, inputShapeTuple.1, inputShapeTuple.2]
        } else {
            fatalError("Could not extract out elements of shape")
        }

        let featureCount = inputShapeArray[(inputShapeArray.count + axis) % inputShapeArray.count]
        return (BatchNorm<Scalar>(featureCount: featureCount, axis: axis, momentum: momentum, epsilon: epsilon), inputShape)
    }
}
