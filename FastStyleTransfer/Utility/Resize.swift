import TensorFlow

/// Resizes input images according to `scaleFactor`.
///
/// Expected input layout: BxHxWxC.
@differentiable(wrt: input, vjp: _vjpResizeNearestNeighbor(input:scaleFactor:))
public func resizeNearestNeighbor<Scalar: TensorFlowFloatingPoint>(
    _ input: Tensor<Scalar>, scaleFactor: Float
) -> Tensor<Scalar> {
    let size = Tensor<Int32>(
        shape: [2],
        scalars: [input.shape[1], input.shape[2]].map { Int32(round(Float($0) * scaleFactor)) }
    )
    return Raw.resizeNearestNeighbor(images: input, size: size)
}

@usableFromInline
internal func _vjpResizeNearestNeighbor<Scalar: TensorFlowFloatingPoint>(
    input: Tensor<Scalar>, scaleFactor: Float
) -> (Tensor<Scalar>, (Tensor<Scalar>) -> Tensor<Scalar>) {
    let result = resizeNearestNeighbor(input, scaleFactor: scaleFactor)
    return (result, { v in
        let size = Tensor<Int32>(
            shape: [2],
            scalars: [input.shape[1], input.shape[2]].map { Int32($0) }
        )
        return Raw.resizeNearestNeighborGrad(grads: result, size: size)
    })
}
