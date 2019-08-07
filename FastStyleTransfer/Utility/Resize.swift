import TensorFlow

@differentiable(wrt: input, vjp: _vjpResizeNearestNeighbor(input:scale_factor:))
public func resizeNearestNeighbor<Scalar: TensorFlowFloatingPoint>(_ input: Tensor<Scalar>, scale_factor: Float) -> Tensor<Scalar> {
    let size = Tensor<Int32>(
            shape: [2],
            scalars: [input.shape[1], input.shape[2]].map { Int32(round(Float($0) * scale_factor)) }
    )
    return Raw.resizeNearestNeighbor(images: input, size: size)
}

@usableFromInline
internal func _vjpResizeNearestNeighbor<Scalar: TensorFlowFloatingPoint>(
        input: Tensor<Scalar>, scale_factor: Float
) -> (Tensor<Scalar>, (Tensor<Scalar>) -> Tensor<Scalar>) {
    let result = resizeNearestNeighbor(input, scale_factor: scale_factor)
    return (result, { v in
        let size = Tensor<Int32>(
                shape: [2],
                scalars: [input.shape[1], input.shape[2]].map { Int32($0) }
        )
        return Raw.resizeNearestNeighborGrad(grads: result, size: size)
    })
}