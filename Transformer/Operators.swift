import TensorFlow

// Contains operators needed for Transformer; these will likely be upstreamed into swift-apis

/// Computes the Gaussian error linear unit (GELU) nonlinear activation function
@differentiable
func gelu<Scalar: TensorFlowFloatingPoint>(_ x: Tensor<Scalar>) -> Tensor<Scalar> {
    let polynomial = 0.79788456 * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + tanh(polynomial))
}

/// Performs batched matrix multiplication of two tensors. The last two axes of each tensor
/// are treated as the matrix axes; all others are treated as batch axes.
@differentiable(
    wrt: (left, right),
    vjp: _vjpBatchedMatmul
    where Scalar : Differentiable & FloatingPoint
)
public func batchedMatmul<Scalar : Numeric>(
    _ left: Tensor<Scalar>,
    _ right: Tensor<Scalar>,
    adjointLeft: Bool = false,
    adjointRight: Bool = false
) -> Tensor<Scalar> {
    return Raw.batchMatMul(left, right, adjX: adjointLeft, adjY: adjointRight)
}

@usableFromInline
func _vjpBatchedMatmul<Scalar : Differentiable & FloatingPoint>(
    _ left: Tensor<Scalar>,
    _ right: Tensor<Scalar>,
    adjointLeft: Bool,
    adjointRight: Bool
) -> (Tensor<Scalar>, (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
    let value = batchedMatmul(left, right, adjointLeft: adjointLeft, adjointRight: adjointRight)
    return (value, { v in
        if !adjointLeft {
            if !adjointRight {
                return (
                    batchedMatmul(v, right, adjointLeft: false, adjointRight: true),
                    batchedMatmul(left, v, adjointLeft: true, adjointRight: false))
            } else {
                return (
                    batchedMatmul(v, right, adjointLeft: false, adjointRight: false),
                    batchedMatmul(v, left, adjointLeft: true, adjointRight: false))
            }
        } else {
            if !adjointRight {
                return (
                    batchedMatmul(right, v, adjointLeft: false, adjointRight: true),
                    batchedMatmul(left, v, adjointLeft: false, adjointRight: false))
            } else {
                return (
                    batchedMatmul(right, v, adjointLeft: true, adjointRight: true),
                    batchedMatmul(v, left, adjointLeft: true, adjointRight: true))
            }
        }
    })
}

public extension Tensor
    where Scalar: TensorFlowFloatingPoint {
    /// Gathers slices of self at the specified indices along the first axis. The result has the
    /// same size in the first axis as the scalar count of the index tensor, and the same
    /// size in subsequent axes as self.
    @differentiable(wrt: self, vjp: _vjpGathering)
    func gathering(atIndices indices: Tensor<Int32>) -> Tensor {
        return Raw.gather(params: self, indices: indices)
    }

    func _vjpGathering(atIndices indices: Tensor<Int32>) -> (Tensor, (Tensor) -> Tensor) {
        let value = gathering(atIndices: indices)
        return (value, { [wShape = shape] seed in
            var valuesShape = wShape
            valuesShape[0] = indices.scalarCount
            let values = seed.reshaped(to: valuesShape)
            let indices = indices.reshaped(to: [indices.scalarCount])
            // TODO provide an option for sparse embedding gradients (e.g. equivalent of Python
            // IndexedSlices)
            return Raw.unsortedSegmentSum(
                data: values,
                segmentIds: indices,
                numSegments: Tensor<Int32>(Int32(wShape[0])))
        })
    }
}
