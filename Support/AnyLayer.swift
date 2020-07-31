import TensorFlow
import _Differentiation

// begin modified copy of https://gist.github.com/dan-zheng/be090293ecea27ce0ad96d769e4a6fbc
internal class _AnyLayerBox<Input: Differentiable, Output: Differentiable, F: FloatingPoint & VectorProtocol & ElementaryFunctions>
where F.VectorSpaceScalar == F {
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
  func _unboxed<U: Layer>(to type: U.Type) -> U?
  where U.TangentVector.VectorSpaceScalar == F {
    fatalError("Must implement")
  }

  func copyToDevice(to device: Device) -> _AnyLayerBox {
    fatalError("Must implement")
  }
}

internal class _ConcreteLayerBox<T: Layer>: _AnyLayerBox<T.Input, T.Output, T.TangentVector.VectorSpaceScalar>
where T.TangentVector.VectorSpaceScalar: FloatingPoint & VectorProtocol & ElementaryFunctions, T.TangentVector.VectorSpaceScalar == T.TangentVector.VectorSpaceScalar.VectorSpaceScalar {
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

  override func _unboxed<U: Layer>(to type: U.Type) -> U?
  where U.TangentVector.VectorSpaceScalar == T.TangentVector.VectorSpaceScalar {
    return (self as? _ConcreteLayerBox<U>)?._base
  }

  override func copyToDevice(to device: Device) -> _AnyLayerBox<T.Input, T.Output, T.TangentVector.VectorSpaceScalar> {
    return _ConcreteLayerBox(T(copying: _base, to: device))
  }

  override func _move(along direction: AnyLayerTangentVector<T.TangentVector.VectorSpaceScalar>) {
    guard
      let directionBase =
        direction.base as? T.TangentVector
    else {
      fatalError("Unexpected tangent vector type")
    }
    _base.move(along: directionBase)
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

public struct AnyLayer<Input: Differentiable, Output: Differentiable, F: FloatingPoint & VectorProtocol & ElementaryFunctions>: Layer, CopyableToDevice
where F.VectorSpaceScalar == F {
  internal var _box: _AnyLayerBox<Input, Output, F>

  internal init(_box: _AnyLayerBox<Input, Output, F>) {
    self._box = _box
  }

  /// The underlying base value.
  public var base: Any {
    return _box._typeErasedBase
  }

  /// Creates a type-erased derivative from the given derivative.
  @differentiable
  public init<T: Layer>(_ base: T)
  where T.Input == Input, T.Output == Output, T.TangentVector.VectorSpaceScalar == F {
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
    return (AnyLayer<Input, Output, F>(base), { v in v.base as! T.TangentVector })
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

internal class _AnyLayerTangentVectorBox<F: FloatingPoint & VectorProtocol & ElementaryFunctions>
where F.VectorSpaceScalar == F {
  // `Equatable` requirements (implied by `AdditiveArithmetic`).
  func _isEqual(to other: _AnyLayerTangentVectorBox) -> Bool {
    fatalError("Must implement")
  }
  func _isNotEqual(to other: _AnyLayerTangentVectorBox) -> Bool {
    fatalError("Must implement")
  }

  func _add(_ x: _AnyLayerTangentVectorBox) -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _subtract(_ x: _AnyLayerTangentVectorBox) -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }

  // `AdditiveArithmetic` requirements.
  class var _zero: _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  
  func _adding(_ x: F) -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _subtracting(_ x: F) -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _scaled(by: F) -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }

  // `Differentiable` requirements.
  func _move(along direction: _AnyLayerTangentVectorBox) {
    fatalError("Must implement")
  }

  // `EuclideanDifferentiable` requirements.
  var _differentiableVectorView: _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }

  // `PointwiseMultiplicative` requirements.
  class var _one: _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  
  func _reciprocal() -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }

  func _pointwiseMultiply(by: _AnyLayerTangentVectorBox) -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }

  // `ElementaryFunctions` requirements.
  func _sqrt() -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _cos() -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _sin() -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _tan() -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _acos() -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _asin() -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _atan() -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _cosh() -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _sinh() -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _tanh() -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _acosh() -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _asinh() -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _atanh() -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _exp() -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _exp2() -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _exp10() -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _expm1() -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _log() -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _log2() -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _log10() -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _log1p() -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _pow(_ y: _AnyLayerTangentVectorBox) -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _pow(_ n: Int) -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _root(_ n: Int) -> _AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }

  /// The underlying base value, type-erased to `Any`.
  var _typeErasedBase: Any {
    fatalError("Must implement")
  }

  /// Returns the underlying value unboxed to the given type, if possible.
  func _unboxed<U: Differentiable & VectorProtocol & ElementaryFunctions & PointwiseMultiplicative>(to type: U.Type) -> U?
    where U.TangentVector == U, U.VectorSpaceScalar == F {
    fatalError("Must implement")
  }
}

extension _AnyLayerTangentVectorBox {
  /// Returns true if the underlying value has type `AnyLayerTangentVector.OpaqueZero`.
  func _getOpaqueScalar() -> F? {
    return _unboxed(to: AnyLayerTangentVector<F>.OpaqueScalar.self)?.value
  }
}

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

internal class _ConcreteAnyLayerTangentVectorBox<T: Differentiable & VectorProtocol & ElementaryFunctions & PointwiseMultiplicative> : _AnyLayerTangentVectorBox<T.VectorSpaceScalar>
  where T.TangentVector == T, T.VectorSpaceScalar: FloatingPoint & VectorProtocol & ElementaryFunctions, T.VectorSpaceScalar.VectorSpaceScalar == T.VectorSpaceScalar
{
  /// The underlying base value.
  var _base: T

  init(_ base: T) {
    self._base = base
  }

  /// The underlying base value, type-erased to `Any`.
  override var _typeErasedBase: Any {
    return _base
  }

  override func _unboxed<U: Differentiable & VectorProtocol & ElementaryFunctions & PointwiseMultiplicative>(to type: U.Type) -> U?
    where U.TangentVector == U, U.VectorSpaceScalar == T.VectorSpaceScalar, U.VectorSpaceScalar : VectorProtocol
  {
    return (self as? _ConcreteAnyLayerTangentVectorBox<U>)?._base
  }

  // `Equatable` requirements (implied by `AdditiveArithmetic`).

  override func _isEqual(to other: _AnyLayerTangentVectorBox<T.VectorSpaceScalar>) -> Bool {
    return _base == other._unboxed(to: T.self)
  }

  override func _isNotEqual(to other: _AnyLayerTangentVectorBox<T.VectorSpaceScalar>) -> Bool {
    return _base != other._unboxed(to: T.self)
  }

  override func _add(_ x: _AnyLayerTangentVectorBox<T.VectorSpaceScalar>) -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    // C + x = x + C
    if let scalar = _getOpaqueScalar() {
      return x._adding(scalar)
    }
    // self + C = self + C
    if let scalar = x._getOpaqueScalar() {
      return self._adding(scalar)
    }

    guard let xBase = x._unboxed(to: T.self) else {
      _derivativeTypeMismatch(T.self, type(of: x._typeErasedBase))
    }
    return _ConcreteAnyLayerTangentVectorBox(_base + xBase)
  }

  override func _subtract(_ x: _AnyLayerTangentVectorBox<T.VectorSpaceScalar>) -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    // C - x = -x + C
    if let scalar = _getOpaqueScalar() {
      return type(of: x)._zero._subtract(x)._adding(scalar)
    }
    // self - C = self - C
    if let scalar = x._getOpaqueScalar() {
      return self._subtracting(scalar)
    }

    guard let xBase = x._unboxed(to: T.self) else {
      _derivativeTypeMismatch(T.self, type(of: x._typeErasedBase))
    }
    return _ConcreteAnyLayerTangentVectorBox(_base - xBase)
  }

  // `AdditiveArithmetic` requirements.
  override class var _zero: _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox(T.zero)
  }
  
  override func _adding(_ x: T.VectorSpaceScalar) -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(_base.adding(x))
  }
  override func _subtracting(_ x: T.VectorSpaceScalar) -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(_base.subtracting(x))
  }
  override func _scaled(by: T.VectorSpaceScalar) -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(_base.scaled(by: by))
  }

  // `PointwiseMultiplicative` requirements.
  override func _reciprocal() -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(_base.reciprocal)
  }

  override class var _one: _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(T.one)
  }

  override func _pointwiseMultiply(by: _AnyLayerTangentVectorBox<T.VectorSpaceScalar>) -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(_base .* by._unboxed(to: T.self)!)
  }

  // `ElementaryFunctions` requirements.
  override func _sqrt() -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(T.sqrt(_base));
  }
  override func _cos() -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(T.cos(_base));
  }
  override func _sin() -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(T.sin(_base));
  }
  override func _tan() -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(T.tan(_base));
  }
  override func _acos() -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(T.acos(_base));
  }
  override func _asin() -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(T.asin(_base));
  }
  override func _atan() -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(T.atan(_base));
  }
  override func _cosh() -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(T.cosh(_base));
  }
  override func _sinh() -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(T.sinh(_base));
  }
  override func _tanh() -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(T.tanh(_base));
  }
  override func _acosh() -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(T.acosh(_base));
  }
  override func _asinh() -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(T.asinh(_base));
  }
  override func _atanh() -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(T.atanh(_base));
  }
  override func _exp() -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(T.exp(_base));
  }
  override func _exp2() -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(T.exp2(_base));
  }
  override func _exp10() -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(T.exp10(_base));
  }
  override func _expm1() -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(T.expm1(_base));
  }
  override func _log() -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(T.log(_base));
  }
  override func _log2() -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(T.log2(_base));
  }
  override func _log10() -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(T.log10(_base));
  }
  override func _log1p() -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(T.log1p(_base));
  }
  override func _pow(_ y: _AnyLayerTangentVectorBox<T.VectorSpaceScalar>) -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(T.pow(_base, y._unboxed(to: T.self)!));
  }
  override func _pow(_ n: Int) -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(T.pow(_base, n));
  }
  override func _root(_ n: Int) -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(T.root(_base, n));
  }

  // `Differentiable` requirements.
  override func _move(along direction: _AnyLayerTangentVectorBox<T.VectorSpaceScalar>) {
    if let scalarDirection = direction._getOpaqueScalar() {
      _base.move(along: T.TangentVector.zero.adding(scalarDirection))
    } else {
      guard let directionBase =
        direction._unboxed(to: T.TangentVector.self) else {
        _derivativeTypeMismatch(T.self, type(of: direction._typeErasedBase))
      }
      _base.move(along: directionBase)
    }
  }

  // `EuclideanDifferentiable` requirements.
  override var _differentiableVectorView: _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return self
  }
}

/// A type-erased derivative value.
///
/// The `AnyLayerTangentVector` type forwards its operations to an arbitrary underlying
/// base derivative value conforming to `Differentiable` and
/// `AdditiveArithmetic`, hiding the specifics of the underlying value.
// public struct AnyLayerTangentVector : EuclideanDifferentiable & AdditiveArithmetic {
public struct AnyLayerTangentVector<F: FloatingPoint & VectorProtocol & ElementaryFunctions>: VectorProtocol & ElementaryFunctions & PointwiseMultiplicative & KeyPathIterable & EuclideanDifferentiable & AdditiveArithmetic
where F.VectorSpaceScalar == F {
  internal var _box: _AnyLayerTangentVectorBox<F>

  internal init(_box: _AnyLayerTangentVectorBox<F>) {
    self._box = _box
  }

  /// The underlying base value.
  public var base: Any {
    return _box._typeErasedBase
  }

  /// Creates a type-erased derivative from the given derivative.
  @differentiable
  public init<T: Differentiable & VectorProtocol & ElementaryFunctions & PointwiseMultiplicative>(_ base: T) where T.TangentVector == T, T.VectorSpaceScalar == F {
    self._box = _ConcreteAnyLayerTangentVectorBox<T>(base)
  }

  @derivative(of: init)
  @usableFromInline
  internal static func _vjpInit<T: Differentiable & VectorProtocol & ElementaryFunctions & PointwiseMultiplicative>(
    _ base: T
  ) -> (value: AnyLayerTangentVector<F>, pullback: (AnyLayerTangentVector<F>) -> T.TangentVector)
    where T.TangentVector == T, T.VectorSpaceScalar == F
  {
    return (AnyLayerTangentVector<F>(base), { v in v.base as! T.TangentVector })
  }

  @derivative(of: init)
  @usableFromInline
  internal static func _jvpInit<T: Differentiable & VectorProtocol & ElementaryFunctions & PointwiseMultiplicative>(
    _ base: T
  ) -> (value: AnyLayerTangentVector<F>, differential: (T.TangentVector) -> AnyLayerTangentVector<F>)
    where T.TangentVector == T, T.VectorSpaceScalar == F
  {
    return (AnyLayerTangentVector<F>(base), { dbase in AnyLayerTangentVector<F>(dbase) })
  }

  public typealias TangentVector = AnyLayerTangentVector

  // `Equatable` requirements (implied by `AdditiveArithmetic`).
  public static func == (lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector) -> Bool {
    return lhs._box._isEqual(to: rhs._box)
  }
  public static func != (lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector) -> Bool {
    return lhs._box._isNotEqual(to: rhs._box)
  }

  // `AdditiveArithmetic` requirements.

  /// Internal struct representing an opaque zero value.
  @frozen
  @usableFromInline
  internal struct OpaqueScalar : EuclideanDifferentiable & AdditiveArithmetic & VectorProtocol & ElementaryFunctions & PointwiseMultiplicative {
    @usableFromInline typealias VectorSpaceScalar = F
    let value: F

    @usableFromInline typealias TangentVector = OpaqueScalar

    init(_ value: F) {
      self.value = value
    }

    @usableFromInline func adding(_ x: F) -> OpaqueScalar {
      return OpaqueScalar(value.adding(x))
    }

    @usableFromInline func subtracting(_ x: F) -> OpaqueScalar {
      return OpaqueScalar(value.subtracting(x))
    }

    @usableFromInline func scaled(by: F) -> OpaqueScalar {
      return OpaqueScalar(value.scaled(by: by))
    }

    // `PointwiseMultiplicative` requirements.
    @usableFromInline static var one: OpaqueScalar {
      return OpaqueScalar(F(exactly: 1)!)
    }

    @usableFromInline var reciprocal: OpaqueScalar {
      return OpaqueScalar(F(exactly: 1)! / value)
    }

    @usableFromInline static func .* (lhs: OpaqueScalar, rhs: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(lhs.value * rhs.value)
    }

    // `ElementaryFunctions` requirements.
    @usableFromInline static func sqrt(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(F.sqrt(x.value))
    }

    @usableFromInline static func cos(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(F.cos(x.value))
    }

    @usableFromInline static func sin(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(F.sin(x.value))
    }

    @usableFromInline static func tan(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(F.tan(x.value))
    }

    @usableFromInline static func acos(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(F.acos(x.value))
    }

    @usableFromInline static func asin(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(F.asin(x.value))
    }

    @usableFromInline static func atan(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(F.atan(x.value))
    }

    @usableFromInline static func cosh(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(F.cosh(x.value))
    }

    @usableFromInline static func sinh(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(F.sinh(x.value))
    }

    @usableFromInline static func tanh(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(F.tanh(x.value))
    }

    @usableFromInline static func acosh(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(F.acosh(x.value))
    }

    @usableFromInline static func asinh(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(F.asinh(x.value))
    }

    @usableFromInline static func atanh(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(F.atanh(x.value))
    }

    @usableFromInline static func exp(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(F.exp(x.value))
    }

    @usableFromInline static func exp2(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(F.exp2(x.value))
    }

    @usableFromInline static func exp10(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(F.exp10(x.value))
    }

    @usableFromInline static func expm1(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(F.expm1(x.value))
    }

    @usableFromInline static func log(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(F.log(x.value))
    }

    @usableFromInline static func log2(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(F.log2(x.value))
    }

    @usableFromInline static func log10(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(F.log10(x.value))
    }

    @usableFromInline static func log1p(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(F.log1p(x.value))
    }

    @usableFromInline static func pow(_ x: OpaqueScalar, _ y: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(F.pow(x.value, y.value))
    }

    @usableFromInline static func pow(_ x: OpaqueScalar, _ n: Int) -> OpaqueScalar {
      return OpaqueScalar(F.pow(x.value, n))
    }

    @usableFromInline static func root(_ x: OpaqueScalar, _ n: Int) -> OpaqueScalar {
      return OpaqueScalar(F.root(x.value, n))
    }
  }

  public static var zero: AnyLayerTangentVector {
    return AnyLayerTangentVector(
      _box: _ConcreteAnyLayerTangentVectorBox<OpaqueScalar>._zero)
  }

  public static func + (
    lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector
  ) -> AnyLayerTangentVector {
    return AnyLayerTangentVector(_box: lhs._box._add(rhs._box))
  }

  @derivative(of: +)
  @usableFromInline internal static func _vjpAdd(
    lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector
  ) -> (value: AnyLayerTangentVector,
        pullback: (AnyLayerTangentVector) -> (AnyLayerTangentVector, AnyLayerTangentVector)) {
    return (lhs + rhs, { v in (v, v) })
  }

  @derivative(of: +)
  @usableFromInline internal static func _jvpAdd(
    lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector
  ) -> (value: AnyLayerTangentVector,
    differential: (AnyLayerTangentVector, AnyLayerTangentVector) -> (AnyLayerTangentVector)) {
      return (lhs + rhs, { (dlhs, drhs) in dlhs + drhs })
  }

  public static func - (
    lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector
  ) -> AnyLayerTangentVector {
    return AnyLayerTangentVector(_box: lhs._box._subtract(rhs._box))
  }

  @derivative(of: -)
  @usableFromInline internal static func _vjpSubtract(
    lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector
  ) -> (value: AnyLayerTangentVector,
        pullback: (AnyLayerTangentVector) -> (AnyLayerTangentVector, AnyLayerTangentVector)) {
    return (lhs - rhs, { v in (v, .zero - v) })
  }

  @derivative(of: -)
  @usableFromInline internal static func _jvpSubtract(
    lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector
  ) -> (value: AnyLayerTangentVector,
        differential: (AnyLayerTangentVector, AnyLayerTangentVector) -> AnyLayerTangentVector) {
    return (lhs - rhs, { (dlhs, drhs) in dlhs - drhs })
  }

  // `VectorProtocol` requirements.
  public static var one: AnyLayerTangentVector {
    return AnyLayerTangentVector(_box: _ConcreteAnyLayerTangentVectorBox<OpaqueScalar>._one)
  }

  public var reciprocal: AnyLayerTangentVector {
    return AnyLayerTangentVector(_box: _box._reciprocal())
  }

  public static func .* (lhs: Self, rhs: Self) -> Self {
    return AnyLayerTangentVector(_box: lhs._box._pointwiseMultiply(by: rhs._box))
  }

  public typealias VectorSpaceScalar = F

  // `AdditiveArithmetic` requirements.
  public func adding(_ x: VectorSpaceScalar) -> Self {
    return AnyLayerTangentVector(_box: _box._adding(x));
  }

  public func subtracting(_ x: VectorSpaceScalar) -> Self {
    return AnyLayerTangentVector(_box: _box._subtracting(x));
  }

  public func scaled(by scalar: VectorSpaceScalar) -> Self {
    return AnyLayerTangentVector(_box: _box._scaled(by: scalar))
  }

  // `ElementaryFunctions` requirements.
  public static func sqrt(_ x: Self) -> Self {
    return AnyLayerTangentVector(_box: x._box._sqrt())
  }
  public static func cos(_ x: Self) -> Self {
    return AnyLayerTangentVector(_box: x._box._cos())
  }
  public static func sin(_ x: Self) -> Self {
    return AnyLayerTangentVector(_box: x._box._sin())
  }
  public static func tan(_ x: Self) -> Self {
    return AnyLayerTangentVector(_box: x._box._tan())
  }
  public static func acos(_ x: Self) -> Self {
    return AnyLayerTangentVector(_box: x._box._acos())
  }
  public static func asin(_ x: Self) -> Self {
    return AnyLayerTangentVector(_box: x._box._asin())
  }
  public static func atan(_ x: Self) -> Self {
    return AnyLayerTangentVector(_box: x._box._atan())
  }
  public static func cosh(_ x: Self) -> Self {
    return AnyLayerTangentVector(_box: x._box._cosh())
  }
  public static func sinh(_ x: Self) -> Self {
    return AnyLayerTangentVector(_box: x._box._sinh())
  }
  public static func tanh(_ x: Self) -> Self {
    return AnyLayerTangentVector(_box: x._box._tanh())
  }
  public static func acosh(_ x: Self) -> Self {
    return AnyLayerTangentVector(_box: x._box._acosh())
  }
  public static func asinh(_ x: Self) -> Self {
    return AnyLayerTangentVector(_box: x._box._asinh())
  }
  public static func atanh(_ x: Self) -> Self {
    return AnyLayerTangentVector(_box: x._box._atanh())
  }
  public static func exp(_ x: Self) -> Self {
    return AnyLayerTangentVector(_box: x._box._exp())
  }
  public static func exp2(_ x: Self) -> Self {
    return AnyLayerTangentVector(_box: x._box._exp2())
  }
  public static func exp10(_ x: Self) -> Self {
    return AnyLayerTangentVector(_box: x._box._exp10())
  }
  public static func expm1(_ x: Self) -> Self {
    return AnyLayerTangentVector(_box: x._box._expm1())
  }
  public static func log(_ x: Self) -> Self {
    return AnyLayerTangentVector(_box: x._box._log())
  }
  public static func log2(_ x: Self) -> Self {
    return AnyLayerTangentVector(_box: x._box._log2())
  }
  public static func log10(_ x: Self) -> Self {
    return AnyLayerTangentVector(_box: x._box._log10())
  }
  public static func log1p(_ x: Self) -> Self {
    return AnyLayerTangentVector(_box: x._box._log1p())
  }
  public static func pow(_ x: Self, _ y: Self) -> Self {
    return AnyLayerTangentVector(_box: x._box._pow(y._box))
  }
  public static func pow(_ x: Self, _ n: Int) -> Self {
    return AnyLayerTangentVector(_box: x._box._pow(n))
  }
  public static func root(_ x: Self, _ n: Int) -> Self {
    return AnyLayerTangentVector(_box: x._box._root(n))
  }

  // `Differentiable` requirements.
  public mutating func move(along direction: TangentVector) {
    _box._move(along: direction._box)
  }

  // `EuclideanDifferentiable` requirements.
  public var differentiableVectorView: TangentVector {
    return self
  }
}

public struct MyLayer: Layer {
    public var conv1 = Conv2D<Float>(filterShape: (5, 5, 1, 6), padding: .same, activation: relu)
    public var pool1 = AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    public var conv2 = Conv2D<Float>(filterShape: (5, 5, 6, 16), activation: relu)
    public var pool2 = AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    public var flatten = Flatten<Float>()
    public var fc1 = Dense<Float>(inputSize: 400, outputSize: 120, activation: relu)
    public var fc2 = Dense<Float>(inputSize: 120, outputSize: 84, activation: relu)
    public var fc3 = Dense<Float>(inputSize: 84, outputSize: 10)

    public init() {}

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let convolved = input.sequenced(through: conv1, pool1, conv2, pool2)
        return convolved.sequenced(through: flatten, fc1, fc2, fc3)
    }
}

public func testFunc() {
  let erased = AnyLayer<Tensor<Float>, Tensor<Float>, Float>(MyLayer())
}
