import TensorFlow
import _Differentiation

/// A helper that raises an error when an erased derivative type does not match up with the true
/// underlying type.
@inline(never)
@usableFromInline
internal func derivativeTypeMismatch(
  _ x: Any.Type, _ y: Any.Type, file: StaticString = #file, line: UInt = #line
) -> Never {
  preconditionFailure("""
    Derivative type mismatch: \
    \(String(reflecting: x)) and \(String(reflecting: y))
    """, file: file, line: line)
}

/// The base type for a type-erased box that encapsulates a layer.
/// Offers forwarders to implement conformance to `Layer` and `CopyableToDevice`.
internal class _AnyLayerBox<Input: Differentiable, Output: Differentiable, F: FloatingPoint & ElementaryFunctions> {
  /// The underlying base layer, type-erased to `Any`.
  var typeErasedBase: Any {
    fatalError("Must implement")
  }

  /// Returns the underlying layer unboxed to the given type, if possible.
  func unboxed<U: Layer>(to type: U.Type) -> U?
  where U.TangentVector.VectorSpaceScalar == F {
    fatalError("Must implement")
  }
  
  // `Differentiable` requirements.
  func _move(along direction: AnyLayerTangentVector<F>) {
    fatalError("Must implement")
  }

  // `EuclideanDifferentiable` requirements.
  var _differentiableVectorView: AnyLayerTangentVector<F> {
    fatalError("Must implement")
  }

  // `Layer` requirements.
  func _callAsFunction(_ input: Input) -> Output {
    fatalError("Must implement")
  }

  func _dCallAsFunction(_ input: Input) -> (value: Output, pullback: (Output.TangentVector) -> (AnyLayerTangentVector<F>, Input.TangentVector)) {
    fatalError("Must implement")
  }

  // `CopyableToDevice` requirements.
  func _copyToDevice(to device: Device) -> _AnyLayerBox {
    fatalError("Must implement")
  }
}

/// A concrete implementation of the type-erased layer wrapper that forwards to an underlying layer.
internal class _ConcreteLayerBox<T: Layer>: _AnyLayerBox<T.Input, T.Output, T.TangentVector.VectorSpaceScalar>
where T.TangentVector.VectorSpaceScalar: FloatingPoint & ElementaryFunctions {
  /// The underlying base value.
  var base: T

  init(_ base: T) {
    self.base = base
  }

  /// The underlying base layer, type-erased to `Any`.
  override var typeErasedBase: Any {
    return base
  }

  /// Returns the underlying layer unboxed to the given type, if possible.
  override func unboxed<U: Layer>(to type: U.Type) -> U?
  where U.TangentVector.VectorSpaceScalar == T.TangentVector.VectorSpaceScalar {
    return (self as? _ConcreteLayerBox<U>)?.base
  }

  // `Differentiable` requirements.
  override func _move(along direction: AnyLayerTangentVector<T.TangentVector.VectorSpaceScalar>) {
    if let scalarDirection = direction.box.getOpaqueScalar() {
      base.move(along: T.TangentVector.zero.adding(scalarDirection))
    } else {
      guard let directionBase =
        direction.unboxed(as: T.TangentVector.self) else {
        derivativeTypeMismatch(T.self, type(of: direction.box.typeErasedBase))
      }
      base.move(along: directionBase)
    }
  }

  // `EuclideanDifferentiable` requirements.
  public override var _differentiableVectorView: AnyLayerTangentVector<T.TangentVector.VectorSpaceScalar> {
    return AnyLayerTangentVector(base.differentiableVectorView)
  }

  // `Layer` requirements.
  override func _callAsFunction(_ input: T.Input) -> T.Output {
    return base.callAsFunction(input)
  }

  // A helper to group together the model an input since we need a pullback with respect to both.
  struct ModelAndInput: Differentiable {
    var model: T
    var input: T.Input
  }

  override func _dCallAsFunction(_ input: T.Input) -> (value: T.Output, pullback: (T.Output.TangentVector) -> (AnyLayerTangentVector<T.TangentVector.VectorSpaceScalar>, T.Input.TangentVector)) {
    let basePullback = valueWithPullback(at: ModelAndInput(model: base, input: input), in: { pair in pair.model.callAsFunction(pair.input) })
    return (
      value: basePullback.value,
      pullback: { (outTangent) in
        let pairTangent = basePullback.pullback(outTangent)
        return (
          AnyLayerTangentVector<T.TangentVector.VectorSpaceScalar>(pairTangent.model),
          pairTangent.input
        )
      }
    )
  }

  // `CopyableToDevice` requirements.
  override func _copyToDevice(to device: Device) -> _AnyLayerBox<T.Input, T.Output, T.TangentVector.VectorSpaceScalar> {
    return _ConcreteLayerBox(T(copying: base, to: device))
  }
}

/// A type-erased layer.
///
/// The `AnyLayer` type forwards its operations to an arbitrary underlying
/// base value conforming to `Layer`, hiding the specifics of the underlying value.
///
/// This erased layer does not implement `KeyPathIterable` due to a Swift constraint that makes it impossible to
/// cast within a keypath (necessary because the layer is stored as an erased `Any` value). The layer _does_ support
/// `CopyableToDevice`, however, so it can be moved between devices.
///
/// The tangent vector of this type is also type-erased, using the `AnyLayerTangentVector` type. All tangents
/// (other than `zero` and `one`) wrap the tangent vector type of the underlying layer.
public struct AnyLayer<Input: Differentiable, Output: Differentiable, F: FloatingPoint & ElementaryFunctions>: CopyableToDevice {
  internal var box: _AnyLayerBox<Input, Output, F>

  internal init(box: _AnyLayerBox<Input, Output, F>) {
    self.box = box
  }

  /// The underlying base layer.
  public var base: Any {
    return box.typeErasedBase
  }

  /// Creates a type-erased derivative from the given layer.
  @differentiable
  public init<T: Layer>(_ base: T)
  where T.Input == Input, T.Output == Output, T.TangentVector.VectorSpaceScalar == F {
    self.box = _ConcreteLayerBox<T>(base)
  }

  public init(copying other: AnyLayer, to device: Device) {
    self.box = other.box._copyToDevice(to: device)
  }

  @inlinable
  @derivative(of: init)
  internal static func _vjpInit<T: Layer>(
    _ base: T
  ) -> (value: AnyLayer, pullback: (AnyLayerTangentVector<F>) -> T.TangentVector)
  where T.Input == Input, T.Output == Output, T.TangentVector.VectorSpaceScalar == F
  {
    return (AnyLayer<Input, Output, F>(base), { v in v.unboxed(as: T.TangentVector.self)! })
  }

  @inlinable
  @derivative(of: init)
  internal static func _jvpInit<T: Layer>(
    _ base: T
  ) -> (
    value: AnyLayer, differential: (T.TangentVector) -> AnyLayerTangentVector<F>
  ) where T.Input == Input, T.Output == Output, T.TangentVector.VectorSpaceScalar == F {
    return (AnyLayer<Input, Output, F>(base), { dbase in AnyLayerTangentVector<F>(dbase) })
  }
}

extension AnyLayer: Differentiable {
  public typealias TangentVector = AnyLayerTangentVector<F>

  public mutating func move(along direction: TangentVector) {
    box._move(along: direction)
  }
}

extension AnyLayer: EuclideanDifferentiable {
  public var differentiableVectorView: TangentVector {
    return box._differentiableVectorView
  }
}

extension AnyLayer: Layer {
  // Must be separate since we have a custom derivative
  func _callAsFunction(_ input: Input) -> Output {
    return box._callAsFunction(input)
  }

  @derivative(of: _callAsFunction)
  func _dCallAsFunction(_ input: Input) -> (value: Output, pullback: (Output.TangentVector) -> (AnyLayerTangentVector<F>, Input.TangentVector)) {
    return box._dCallAsFunction(input)
  }

  @differentiable
  public func callAsFunction(_ input: Input) -> Output {
    return _callAsFunction(input)
  }
}
