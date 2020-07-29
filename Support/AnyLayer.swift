import TensorFlow
import _Differentiation

// begin modified copy of https://gist.github.com/dan-zheng/be090293ecea27ce0ad96d769e4a6fbc
internal class _AnyLayerBox<F: AdditiveArithmetic> {
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

  /// Returns the underlying value unboxed to the given type, if possible.
  func _unboxed<U: Layer>(to type: U.Type) -> U?
  where U.TangentVector.VectorSpaceScalar == F {
    fatalError("Must implement")
  }

  func copyToDevice(to device: Device) -> _AnyLayerBox {
    fatalError("Must implement")
  }
}

internal class _ConcreteLayerBox<T: Layer>: _AnyLayerBox<T.TangentVector.VectorSpaceScalar> {
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

  override func copyToDevice(to device: Device) -> _AnyLayerBox<T.TangentVector.VectorSpaceScalar> {
    return _ConcreteLayerBox(T(copying: _base, to: device))
  }

  override func _move(along direction: AnyLayerTangentVector<T.TangentVector.VectorSpaceScalar>) {
    if !direction._box._isOpaqueZero() {
      guard
        let directionBase =
          direction.base as? T.TangentVector
      else {
        fatalError()
      }
      _base.move(along: directionBase)
    }
  }
}

public struct AnyLayer<F: AdditiveArithmetic>: EuclideanDifferentiable, CopyableToDevice {
  internal var _box: _AnyLayerBox<F>

  internal init(_box: _AnyLayerBox<F>) {
    self._box = _box
  }

  /// The underlying base value.
  public var base: Any {
    return _box._typeErasedBase
  }

  /// Creates a type-erased derivative from the given derivative.
  @differentiable
  public init<T: Layer>(_ base: T) where T.TangentVector.VectorSpaceScalar == F {
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
  where T.TangentVector.VectorSpaceScalar == F
  {
    return (AnyLayer<F>(base), { v in v.base as! T.TangentVector })
  }

  @inlinable
  @derivative(of: init)
  internal static func _jvpInit<T: Layer>(
    _ base: T
  ) -> (
    value: AnyLayer, differential: (T.TangentVector) -> AnyLayerTangentVector<F>
  ) where T.TangentVector.VectorSpaceScalar == F {
    return (AnyLayer<F>(base), { dbase in AnyLayerTangentVector<F>(dbase) })
  }

  public typealias TangentVector = AnyLayerTangentVector<F>

  public var differentiableVectorView: TangentVector {
    return _box._differentiableVectorView
  }

  public mutating func move(along direction: TangentVector) {
    _box._move(along: direction)
  }
}

internal class _AnyLayerTangentVectorBox<F: AdditiveArithmetic> {
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

  /// The underlying base value, type-erased to `Any`.
  var _typeErasedBase: Any {
    fatalError("Must implement")
  }

  /// Returns the underlying value unboxed to the given type, if possible.
  func _unboxed<U>(to type: U.Type) -> U?
    where U : Differentiable & VectorProtocol, U.TangentVector == U, U.VectorSpaceScalar == F {
    fatalError("Must implement")
  }
}

extension _AnyLayerTangentVectorBox {
  /// Returns true if the underlying value has type `AnyLayerTangentVector.OpaqueZero`.
  func _isOpaqueZero() -> Bool {
    return _unboxed(to: AnyLayerTangentVector<F>.OpaqueZero.self) != nil
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

internal class _ConcreteAnyLayerTangentVectorBox<T: Differentiable & VectorProtocol> : _AnyLayerTangentVectorBox<T.VectorSpaceScalar>
  where T.TangentVector == T
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

  override func _unboxed<U: Differentiable & VectorProtocol>(to type: U.Type) -> U?
    where U.TangentVector == U, U.VectorSpaceScalar == T.VectorSpaceScalar
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
    // 0 + x = x
    if _isOpaqueZero() {
      return x
    }
    // y + 0 = y
    if x._isOpaqueZero() {
      return self
    }
    guard let xBase = x._unboxed(to: T.self) else {
      _derivativeTypeMismatch(T.self, type(of: x._typeErasedBase))
    }
    return _ConcreteAnyLayerTangentVectorBox(_base + xBase)
  }

  override func _subtract(_ x: _AnyLayerTangentVectorBox<T.VectorSpaceScalar>) -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    // y - 0 = y
    if x._isOpaqueZero() {
      return self
    }
    // 0 - x = -x
    if _isOpaqueZero() {
      return type(of: x)._zero._subtract(x)
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

  // `ElementaryFunctions` requirements.
  // TODO

  // `Differentiable` requirements.
  override func _move(along direction: _AnyLayerTangentVectorBox<T.VectorSpaceScalar>) {
    if direction._isOpaqueZero() {
      return
    }
    // The case where `self._isOpaqueZero()` returns true is handled in
    // `AnyLayerTangentVector.move(along:)`.
    guard let directionBase =
      direction._unboxed(to: T.TangentVector.self) else {
      _derivativeTypeMismatch(T.self, type(of: direction._typeErasedBase))
    }
    _base.move(along: directionBase)
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
public struct AnyLayerTangentVector<F: AdditiveArithmetic>: VectorProtocol & ElementaryFunctions & PointwiseMultiplicative & KeyPathIterable & EuclideanDifferentiable & AdditiveArithmetic {
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
  public init<T: Differentiable & VectorProtocol>(_ base: T) where T.TangentVector == T, T.VectorSpaceScalar == F {
    self._box = _ConcreteAnyLayerTangentVectorBox<T>(base)
  }

  @derivative(of: init)
  @usableFromInline
  internal static func _vjpInit<T: Differentiable & VectorProtocol>(
    _ base: T
  ) -> (value: AnyLayerTangentVector<F>, pullback: (AnyLayerTangentVector<F>) -> T.TangentVector)
    where T.TangentVector == T, T.VectorSpaceScalar == F
  {
    return (AnyLayerTangentVector<F>(base), { v in v.base as! T.TangentVector })
  }

  @derivative(of: init)
  @usableFromInline
  internal static func _jvpInit<T: Differentiable & VectorProtocol>(
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
  internal struct OpaqueZero : EuclideanDifferentiable & AdditiveArithmetic & VectorProtocol {
    @usableFromInline typealias VectorSpaceScalar = F

    @usableFromInline func adding(_ x: F) -> OpaqueZero {
      fatalError("not implemented")
    }

    @usableFromInline func subtracting(_ x: F) -> OpaqueZero {
      fatalError("not implemented")
    }

    @usableFromInline func scaled(by: F) -> OpaqueZero {
      return self
    }
  }

  public static var zero: AnyLayerTangentVector {
    return AnyLayerTangentVector(
      _box: _ConcreteAnyLayerTangentVectorBox<OpaqueZero>(OpaqueZero.zero))
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
    fatalError()
  }

  public var reciprocal: AnyLayerTangentVector {
    fatalError()
  }

  public static func .* (lhs: Self, rhs: Self) -> Self {
    fatalError()
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
    fatalError()
  }
  public static func cos(_ x: Self) -> Self {
    fatalError()
  }
  public static func sin(_ x: Self) -> Self {
    fatalError()
  }
  public static func tan(_ x: Self) -> Self {
    fatalError()
  }
  public static func acos(_ x: Self) -> Self {
    fatalError()
  }
  public static func asin(_ x: Self) -> Self {
    fatalError()
  }
  public static func atan(_ x: Self) -> Self {
    fatalError()
  }
  public static func cosh(_ x: Self) -> Self {
    fatalError()
  }
  public static func sinh(_ x: Self) -> Self {
    fatalError()
  }
  public static func tanh(_ x: Self) -> Self {
    fatalError()
  }
  public static func acosh(_ x: Self) -> Self {
    fatalError()
  }
  public static func asinh(_ x: Self) -> Self {
    fatalError()
  }
  public static func atanh(_ x: Self) -> Self {
    fatalError()
  }
  public static func exp(_ x: Self) -> Self {
    fatalError()
  }
  public static func exp2(_ x: Self) -> Self {
    fatalError()
  }
  public static func exp10(_ x: Self) -> Self {
    fatalError()
  }
  public static func expm1(_ x: Self) -> Self {
    fatalError()
  }
  public static func log(_ x: Self) -> Self {
    fatalError()
  }
  public static func log2(_ x: Self) -> Self {
    fatalError()
  }
  public static func log10(_ x: Self) -> Self {
    fatalError()
  }
  public static func log1p(_ x: Self) -> Self {
    fatalError()
  }
  public static func pow(_ x: Self, _ y: Self) -> Self {
    fatalError()
  }
  public static func pow(_ x: Self, _ n: Int) -> Self {
    fatalError()
  }
  public static func root(_ x: Self, _ n: Int) -> Self {
    fatalError()
  }

  // `Differentiable` requirements.
  public mutating func move(along direction: TangentVector) {
    if _box._isOpaqueZero() {
      _box = direction._box
      return
    }
    _box._move(along: direction._box)
  }

  // `EuclideanDifferentiable` requirements.
  public var differentiableVectorView: TangentVector {
    return self
  }
}
