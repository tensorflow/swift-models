import TensorFlow
import _Differentiation

@inline(never)
@usableFromInline
internal func _derivativeTypeMismatch(
  _ x: Any.Type, _ y: Any.Type, file: StaticString = #file, line: UInt = #line
) -> Never {
  preconditionFailure("""
    Derivative type mismatch: \
    \(String(reflecting: x)) and \(String(reflecting: y))
    """, file: file, line: line)
}

// TODO(shadaj): docs
// TODO(shadaj): floating and vector???
internal class _AnyLayerBox<Input: Differentiable, Output: Differentiable, F: FloatingPoint & ElementaryFunctions> {
  // `Differentiable` requirements.
  func _move(along direction: AnyLayerTangentVector<F>) {
    fatalError("Must implement")
  }

  /// The underlying base value, type-erased to `Any`.
  var _typeErasedBase: Any {
    fatalError("Must implement")
  }

  var _differentiableVectorView: AnyLayerTangentVector<F> {
    fatalError("Must implement")
  }

  func _callAsFunction(_ input: Input) -> Output {
    fatalError("Must implement")
  }

  func _dCallAsFunction(_ input: Input) -> (value: Output, pullback: (Output.TangentVector) -> (AnyLayerTangentVector<F>, Input.TangentVector)) {
    fatalError("Must implement")
  }

  /// Returns the underlying value unboxed to the given type, if possible.
  func unboxed<U: Layer>(to type: U.Type) -> U?
  where U.TangentVector.VectorSpaceScalar == F {
    fatalError("Must implement")
  }

  func copyToDevice(to device: Device) -> _AnyLayerBox {
    fatalError("Must implement")
  }
}

internal class _ConcreteLayerBox<T: Layer>: _AnyLayerBox<T.Input, T.Output, T.TangentVector.VectorSpaceScalar>
where T.TangentVector.VectorSpaceScalar: FloatingPoint & ElementaryFunctions {
  /// The underlying base value.
  var _base: T

  init(_ base: T) {
    self._base = base
  }

  /// The underlying base value, type-erased to `Any`.
  override var _typeErasedBase: Any {
    return _base
  }

  public override var _differentiableVectorView: AnyLayerTangentVector<T.TangentVector.VectorSpaceScalar> {
    return AnyLayerTangentVector(_base.differentiableVectorView)
  }

  override func unboxed<U: Layer>(to type: U.Type) -> U?
  where U.TangentVector.VectorSpaceScalar == T.TangentVector.VectorSpaceScalar {
    return (self as? _ConcreteLayerBox<U>)?._base
  }

  override func copyToDevice(to device: Device) -> _AnyLayerBox<T.Input, T.Output, T.TangentVector.VectorSpaceScalar> {
    return _ConcreteLayerBox(T(copying: _base, to: device))
  }

  override func _move(along direction: AnyLayerTangentVector<T.TangentVector.VectorSpaceScalar>) {
    if let scalarDirection = direction._box._getOpaqueScalar() {
      _base.move(along: T.TangentVector.zero.adding(scalarDirection))
    } else {
      guard let directionBase =
        direction.unboxed(as: T.TangentVector.self) else {
        _derivativeTypeMismatch(T.self, type(of: direction._box._typeErasedBase))
      }
      _base.move(along: directionBase)
    }
  }

  override func _callAsFunction(_ input: T.Input) -> T.Output {
    return _base.callAsFunction(input)
  }

  struct ModelAndInput: Differentiable {
    var model: T
    var input: T.Input
  }

  override func _dCallAsFunction(_ input: T.Input) -> (value: T.Output, pullback: (T.Output.TangentVector) -> (AnyLayerTangentVector<T.TangentVector.VectorSpaceScalar>, T.Input.TangentVector)) {
    let basePullback = valueWithPullback(at: ModelAndInput(model: _base, input: input), in: { pair in pair.model.callAsFunction(pair.input) })
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
public struct AnyLayer<Input: Differentiable, Output: Differentiable, F: FloatingPoint & ElementaryFunctions>: Layer, CopyableToDevice {
  internal var _box: _AnyLayerBox<Input, Output, F>

  internal init(_box: _AnyLayerBox<Input, Output, F>) {
    self._box = _box
  }

  /// The underlying base layer.
  public var base: Any {
    return _box._typeErasedBase
  }

  /// Creates a type-erased derivative from the given layer.
  @differentiable
  public init<T: Layer>(_ base: T)
  where T.Input == Input, T.Output == Output, T.TangentVector.VectorSpaceScalar == F { // TODO(shadaj): is there a shorter path to T.TangentVector.VectorSpaceScalar?
    self._box = _ConcreteLayerBox<T>(base)
  }

  public init(copying other: AnyLayer, to device: Device) {
    self._box = other._box.copyToDevice(to: device)
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

  public typealias TangentVector = AnyLayerTangentVector<F>

  public var differentiableVectorView: TangentVector {
    return _box._differentiableVectorView
  }

  public mutating func move(along direction: TangentVector) {
    _box._move(along: direction)
  }

  // Must be separate since we have a custom derivative
  func _callAsFunction(_ input: Input) -> Output {
    return _box._callAsFunction(input)
  }

  @derivative(of: _callAsFunction)
  func _dCallAsFunction(_ input: Input) -> (value: Output, pullback: (Output.TangentVector) -> (AnyLayerTangentVector<F>, Input.TangentVector)) {
    return _box._dCallAsFunction(input)
  }

  @differentiable
  public func callAsFunction(_ input: Input) -> Output {
    return _callAsFunction(input)
  }
}

