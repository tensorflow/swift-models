import TensorFlow
import _Differentiation

// begin modified copy of https://gist.github.com/dan-zheng/be090293ecea27ce0ad96d769e4a6fbc
internal protocol _AnyDifferentiableBox {
  // `Differentiable` requirements.
  mutating func _move(along direction: AnyLayerTangentVector)

  /// The underlying base value, type-erased to `Any`.
  var _typeErasedBase: Any { get }

  var _differentiableVectorView: AnyLayerTangentVector { get }

  /// Returns the underlying value unboxed to the given type, if possible.
  func _unboxed<U: EuclideanDifferentiable>(to type: U.Type) -> U?
  where U.TangentVector : VectorProtocol, U.TangentVector.VectorSpaceScalar == Float
}

internal struct _ConcreteDifferentiableBox<T: EuclideanDifferentiable>: _AnyDifferentiableBox
where T.TangentVector: VectorProtocol, T.TangentVector.VectorSpaceScalar == Float
{
  /// The underlying base value.
  var _base: T

  init(_ base: T) {
    self._base = base
  }

  /// The underlying base value, type-erased to `Any`.
  var _typeErasedBase: Any {
    return _base
  }

  public var _differentiableVectorView: AnyLayerTangentVector {
    return AnyLayerTangentVector(_base.differentiableVectorView)
  }

  func _unboxed<U: EuclideanDifferentiable>(to type: U.Type) -> U?
  where U.TangentVector : VectorProtocol, U.TangentVector.VectorSpaceScalar == Float {
    return (self as? _ConcreteDifferentiableBox<U>)?._base
  }

  mutating func _move(along direction: AnyLayerTangentVector) {
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

public struct AnyDifferentiable: EuclideanDifferentiable {
  internal var _box: _AnyDifferentiableBox

  internal init(_box: _AnyDifferentiableBox) {
    self._box = _box
  }

  /// The underlying base value.
  public var base: Any {
    return _box._typeErasedBase
  }

  /// Creates a type-erased derivative from the given derivative.
  @differentiable
  public init<T: EuclideanDifferentiable>(_ base: T) where T.TangentVector : VectorProtocol, T.TangentVector.VectorSpaceScalar == Float {
    self._box = _ConcreteDifferentiableBox<T>(base)
  }

  @inlinable
  @derivative(of: init)
  internal static func _vjpInit<T: EuclideanDifferentiable>(
    _ base: T
  ) -> (value: AnyDifferentiable, pullback: (AnyLayerTangentVector) -> T.TangentVector)
  where T.TangentVector : VectorProtocol, T.TangentVector.VectorSpaceScalar == Float
  {
    return (AnyDifferentiable(base), { v in v.base as! T.TangentVector })
  }

  @inlinable
  @derivative(of: init)
  internal static func _jvpInit<T: EuclideanDifferentiable>(
    _ base: T
  ) -> (
    value: AnyDifferentiable, differential: (T.TangentVector) -> AnyLayerTangentVector
  ) where T.TangentVector : VectorProtocol, T.TangentVector.VectorSpaceScalar == Float {
    return (AnyDifferentiable(base), { dbase in AnyLayerTangentVector(dbase) })
  }

  public typealias TangentVector = AnyLayerTangentVector

  public var differentiableVectorView: TangentVector {
    return _box._differentiableVectorView
  }

  public mutating func move(along direction: TangentVector) {
    _box._move(along: direction)
  }
}

internal protocol _AnyLayerTangentVectorBox {
  // `Equatable` requirements (implied by `AdditiveArithmetic`).
  func _isEqual(to other: _AnyLayerTangentVectorBox) -> Bool
  func _isNotEqual(to other: _AnyLayerTangentVectorBox) -> Bool

  // `AdditiveArithmetic` requirements.
  static var _zero: _AnyLayerTangentVectorBox { get }
  func _adding(_ x: _AnyLayerTangentVectorBox) -> _AnyLayerTangentVectorBox
  func _subtracting(_ x: _AnyLayerTangentVectorBox) -> _AnyLayerTangentVectorBox

  func _scaled(by: Float) -> _AnyLayerTangentVectorBox

  // `Differentiable` requirements.
  mutating func _move(along direction: _AnyLayerTangentVectorBox)

  // `EuclideanDifferentiable` requirements.
  var _differentiableVectorView: _AnyLayerTangentVectorBox { get }

  /// The underlying base value, type-erased to `Any`.
  var _typeErasedBase: Any { get }

  /// Returns the underlying value unboxed to the given type, if possible.
  func _unboxed<U>(to type: U.Type) -> U?
    where U : Differentiable & VectorProtocol, U.TangentVector == U, U.VectorSpaceScalar == Float
}

extension _AnyLayerTangentVectorBox {
  /// Returns true if the underlying value has type `AnyLayerTangentVector.OpaqueZero`.
  func _isOpaqueZero() -> Bool {
    return _unboxed(to: AnyLayerTangentVector.OpaqueZero.self) != nil
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

internal struct _ConcreteAnyLayerTangentVectorBox<T> : _AnyLayerTangentVectorBox
  where T : Differentiable & VectorProtocol, T.TangentVector == T, T.VectorSpaceScalar == Float
{
  /// The underlying base value.
  var _base: T

  init(_ base: T) {
    self._base = base
  }

  /// The underlying base value, type-erased to `Any`.
  var _typeErasedBase: Any {
    return _base
  }

  func _unboxed<U>(to type: U.Type) -> U?
    where U : Differentiable & VectorProtocol, U.TangentVector == U, U.VectorSpaceScalar == Float
  {
    return (self as? _ConcreteAnyLayerTangentVectorBox<U>)?._base
  }

  // `Equatable` requirements (implied by `AdditiveArithmetic`).

  func _isEqual(to other: _AnyLayerTangentVectorBox) -> Bool {
    return _base == other._unboxed(to: T.self)
  }

  func _isNotEqual(to other: _AnyLayerTangentVectorBox) -> Bool {
    return _base != other._unboxed(to: T.self)
  }

  // `AdditiveArithmetic` requirements.

  static var _zero: _AnyLayerTangentVectorBox {
    return _ConcreteAnyLayerTangentVectorBox(T.zero)
  }

  func _adding(_ x: _AnyLayerTangentVectorBox) -> _AnyLayerTangentVectorBox {
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

  func _subtracting(_ x: _AnyLayerTangentVectorBox) -> _AnyLayerTangentVectorBox {
    // y - 0 = y
    if x._isOpaqueZero() {
      return self
    }
    // 0 - x = -x
    if _isOpaqueZero() {
      return type(of: x)._zero._subtracting(x)
    }
    guard let xBase = x._unboxed(to: T.self) else {
      _derivativeTypeMismatch(T.self, type(of: x._typeErasedBase))
    }
    return _ConcreteAnyLayerTangentVectorBox(_base - xBase)
  }

  func _scaled(by: Float) -> _AnyLayerTangentVectorBox {
    return _ConcreteAnyLayerTangentVectorBox(_base.scaled(by: by))
  }

  // `Differentiable` requirements.

  mutating func _move(along direction: _AnyLayerTangentVectorBox) {
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
  var _differentiableVectorView: _AnyLayerTangentVectorBox {
    return self
  }
}

/// A type-erased derivative value.
///
/// The `AnyLayerTangentVector` type forwards its operations to an arbitrary underlying
/// base derivative value conforming to `Differentiable` and
/// `AdditiveArithmetic`, hiding the specifics of the underlying value.
// public struct AnyLayerTangentVector : EuclideanDifferentiable & AdditiveArithmetic {
public struct AnyLayerTangentVector: VectorProtocol & ElementaryFunctions & PointwiseMultiplicative & KeyPathIterable & EuclideanDifferentiable & AdditiveArithmetic {
  internal var _box: _AnyLayerTangentVectorBox

  internal init(_box: _AnyLayerTangentVectorBox) {
    self._box = _box
  }

  /// The underlying base value.
  public var base: Any {
    return _box._typeErasedBase
  }

  /// Creates a type-erased derivative from the given derivative.
  @differentiable
  public init<T>(_ base: T) where T : Differentiable & VectorProtocol, T.TangentVector == T, T.VectorSpaceScalar == Float {
    self._box = _ConcreteAnyLayerTangentVectorBox<T>(base)
  }

  @derivative(of: init)
  @usableFromInline
  internal static func _vjpInit<T>(
    _ base: T
  ) -> (value: AnyLayerTangentVector, pullback: (AnyLayerTangentVector) -> T.TangentVector)
    where T : Differentiable & VectorProtocol, T.TangentVector == T, T.VectorSpaceScalar == Float
  {
    return (AnyLayerTangentVector(base), { v in v.base as! T.TangentVector })
  }

  @derivative(of: init)
  @usableFromInline
  internal static func _jvpInit<T>(
    _ base: T
  ) -> (value: AnyLayerTangentVector, differential: (T.TangentVector) -> AnyLayerTangentVector)
    where T : Differentiable & VectorProtocol, T.TangentVector == T, T.VectorSpaceScalar == Float
  {
    return (AnyLayerTangentVector(base), { dbase in AnyLayerTangentVector(dbase) })
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
    @usableFromInline typealias VectorSpaceScalar = Float

    @usableFromInline func adding(_ x: Float) -> OpaqueZero {
      fatalError("not implemented")
    }

    @usableFromInline func subtracting(_ x: Float) -> OpaqueZero {
      fatalError("not implemented")
    }

    @usableFromInline func scaled(by: Float) -> OpaqueZero {
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
    return AnyLayerTangentVector(_box: lhs._box._adding(rhs._box))
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
    return AnyLayerTangentVector(_box: lhs._box._subtracting(rhs._box))
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

  public typealias VectorSpaceScalar = Float

  public func adding(_ x: VectorSpaceScalar) -> Self {
    fatalError()
  }

  public func subtracting(_ x: VectorSpaceScalar) -> Self {
    fatalError()
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
