import TensorFlow

public struct AutoFlatten<Scalar>: AutoLayer where Scalar: TensorFlowFloatingPoint {
    public typealias InstanceType = Flatten<Scalar>
    public typealias InputShape = (Int, Int, Int)
    public typealias OutputShape = Int

    public init() {}

    public func buildModelWithOutputShape(inputShape: (Int, Int, Int)) -> (InstanceType, Int) {
        return (Flatten<Scalar>(), inputShape.0 * inputShape.1 * inputShape.2)
    }
}
