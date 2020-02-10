import TensorFlow

/// Resizes input images according to `scaleFactor`.
///
/// Expected input layout: BxHxWxC.
@differentiable(wrt: input)
public func resizeNearestNeighbor<Scalar: TensorFlowFloatingPoint>(
    _ input: Tensor<Scalar>, scaleFactor: Float
) -> Tensor<Scalar> {
    let size = Tensor<Int32>(
        shape: [2],
        scalars: [input.shape[1], input.shape[2]].map { Int32(roundf(Float($0) * scaleFactor)) }
    )
    return _Raw.resizeNearestNeighbor(images: input, size: size)
}

@usableFromInline
@derivative(of: resizeNearestNeighbor, wrt: input)
internal func _vjpResizeNearestNeighbor<Scalar: TensorFlowFloatingPoint>(
    input: Tensor<Scalar>, scaleFactor: Float
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> Tensor<Scalar>) {
    let result = resizeNearestNeighbor(input, scaleFactor: scaleFactor)
    return (result, { v in
        let size = Tensor<Int32>(
            shape: [2],
            scalars: [input.shape[1], input.shape[2]].map { Int32($0) }
        )
        return _Raw.resizeNearestNeighborGrad(grads: result, size: size)
    })
}
