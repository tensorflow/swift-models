import TensorFlow
import _Differentiation

struct DynamicLayerStore: EuclideanDifferentiable, KeyPathIterable {
  @differentiable var underlying: AnyDifferentiable

  init<T: Layer>(_ layer: T) where T.TangentVector.VectorSpaceScalar == Float {
    underlying = AnyDifferentiable(_box: _ConcreteDifferentiableBox(layer))
  }

  @differentiable
  func callDynamically<L: Layer>(
    _ input: L.Input,
    layerType: L.Type
  ) -> L.Output where L.TangentVector.VectorSpaceScalar == Float {
    // TODO(shadaj): unsafeDowncast
    return (underlying.base as! L).callAsFunction(input)
  }

  @derivative(of: callDynamically)
  func dCallDynamically<L: Layer>(
    _ input: L.Input,
    layerType: L.Type
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
  typealias CallFunction = @differentiable ([DynamicLayerStore], Tensor<Float>) -> Tensor<Float>

  var layers: [DynamicLayerStore] = []
  @noDerivative var nodeToLayer: [AnyTracingLayer : Int]

  @noDerivative let callFunction: CallFunction

  init(layers: [DynamicLayerStore], nodeToLayer: [AnyTracingLayer : Int], callFunction: @escaping CallFunction) {
    self.layers = layers
    self.nodeToLayer = nodeToLayer
    self.callFunction = callFunction
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    return callFunction(layers, input)
  }

  public subscript<T>(_ key: TracingLayer<T>) -> T {
    return layers[nodeToLayer[key]!].underlying.base as! T
  }
}
