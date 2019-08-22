import TensorFlow
import Foundation

public extension Tensor where Scalar: Numeric {
    /// Returns a padded tensor according to the specified padding sizes.
    @inlinable
    @differentiable(wrt: self, vjp: _vjpPaddedWithReflection(forSizes:)
        where Scalar: TensorFlowFloatingPoint)
    func paddedWithReflection(forSizes sizes: [(before: Int, after: Int)]) -> Tensor {
        let paddings = Tensor<Int32>(
            shape: [sizes.count, 2],
            scalars: sizes.flatMap { [Int32($0.before), Int32($0.after)] }
        )
        return Raw.mirrorPad(self, paddings: paddings, mode: Raw.Mode5.reflect)
    }
}

internal extension Tensor where Scalar: TensorFlowFloatingPoint {
    @inlinable
    func _vjpPaddedWithReflection(
        forSizes sizes: [(before: Int, after: Int)]
    ) -> (Tensor, (Tensor) -> Tensor) {
        let result = paddedWithReflection(forSizes: sizes)
        return (result, { v in
            let paddings = Tensor<Int32>(
                shape: [sizes.count, 2],
                scalars: sizes.flatMap { [Int32($0.before), Int32($0.after)] }
            )
            return Raw.mirrorPadGrad(result, paddings: paddings, mode: Raw.Mode5.reflect)
        })
    }
}
