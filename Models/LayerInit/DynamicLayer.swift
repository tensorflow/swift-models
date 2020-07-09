import TensorFlow
import _Differentiation

public struct DynamicLayerStore: EuclideanDifferentiable, KeyPathIterable {
  @differentiable var underlying: AnyDifferentiable

  public init<T: Layer>(_ layer: T) where T.TangentVector.VectorSpaceScalar == Float {
    underlying = AnyDifferentiable(_box: _ConcreteDifferentiableBox(layer))
  }

  @differentiable
  public func callWithLayer<T: Differentiable, L: Layer, R: Differentiable>(_ input: T, _ thunk: @differentiable (L, T) -> R) -> R where L.TangentVector.VectorSpaceScalar == Float {
    return thunk(underlying.base as! L, input)
  }

  @derivative(of: callWithLayer)
  public func dCallDynamically<T: Differentiable, L: Layer, R: Differentiable>(_ input: T, _ thunk: @differentiable (L, T) -> R) -> (value: R, pullback: (R.TangentVector) -> (TangentVector, T.TangentVector)) where L.TangentVector.VectorSpaceScalar == Float {
    let underlyingLayer = underlying.base as! L
    let (y, pullback) = valueWithPullback(
      at: underlyingLayer, input,
      in: thunk
    )

    return (value: y, pullback: { v in
      let (pullbackUnderlying, inputPullback) = pullback(v)
      return (TangentVector(underlying: AnyLayerTangentVector(pullbackUnderlying)), inputPullback)
    })
  }
}

public struct ComposedLayer: Layer {
  var layers: [DynamicLayerStore] = []
  @noDerivative let callFunction: @differentiable ([DynamicLayerStore], Tensor<Float>) -> Tensor<Float>

  public init(layers: [DynamicLayerStore], callFunction: @escaping @differentiable ([DynamicLayerStore], Tensor<Float>) -> Tensor<Float>) {
    self.layers = layers
    self.callFunction = callFunction
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    return callFunction(layers, input)
  }
}
