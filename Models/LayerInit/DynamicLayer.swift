import TensorFlow
import _Differentiation

public struct DynamicLayerStore: EuclideanDifferentiable, KeyPathIterable {
  @differentiable var underlying: AnyDifferentiable

  public init<T: Layer>(_ layer: T) where T.TangentVector.VectorSpaceScalar == Float {
    underlying = AnyDifferentiable(_box: _ConcreteDifferentiableBox(layer))
  }

  @differentiable
  public func callDynamically<L: Layer>(
    _ input: L.Input,
    layerSpecializer: L?
  ) -> L.Output where L.TangentVector.VectorSpaceScalar == Float {
    return (underlying.base as! L).callAsFunction(input)
  }

  @derivative(of: callDynamically)
  public func dCallDynamically<L: Layer>(
    _ input: L.Input,
    layerSpecializer: L?
  ) -> (
    value: L.Output,
    pullback: (L.Output.TangentVector) -> (TangentVector, L.Input.TangentVector)
  ) where L.TangentVector.VectorSpaceScalar == Float {
    let underlyingLayer = underlying.base as! L
    let (y, pullback) = valueWithPullback(
      at: underlyingLayer, input,
      in: { $0.callAsFunction($1) }
    )

    return (value: y, pullback: { v in
      let (pullbackUnderlying, inputPullback) = pullback(v)
      return (TangentVector(underlying: AnyLayerTangentVector(pullbackUnderlying)), inputPullback)
    })
  }
}

public struct ComposedLayer: Layer {
  public typealias CallFunction = @differentiable ([DynamicLayerStore], Tensor<Float>) -> Tensor<Float>

  var layers: [DynamicLayerStore] = []
  @noDerivative let callFunction: CallFunction

  public init(layers: [DynamicLayerStore], callFunction: @escaping CallFunction) {
    self.layers = layers
    self.callFunction = callFunction
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    return callFunction(layers, input)
  }
}
