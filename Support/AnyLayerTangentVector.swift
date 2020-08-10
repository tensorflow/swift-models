import TensorFlow
import _Differentiation

internal class _AnyLayerTangentVectorBox<F: FloatingPoint & ElementaryFunctions> {
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

  // TODO(shadaj): split out into separate type?
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
  func _unboxed<U: Differentiable & VectorProtocol & ElementaryFunctions & PointwiseMultiplicative>(as type: U.Type) -> U?
    where U.TangentVector == U, U.VectorSpaceScalar == F {
    fatalError("Must implement")
  }
}

extension _AnyLayerTangentVectorBox {
  /// Optionally returns the underlying scalar if the wrapped value has type `AnyLayerTangentVector.OpaqueScalar`.
  func _getOpaqueScalar() -> F? {
    return _unboxed(as: AnyLayerTangentVector<F>.OpaqueScalar.self)?.value
  }
}

internal class _ConcreteAnyLayerTangentVectorBox<T: Differentiable & VectorProtocol & ElementaryFunctions & PointwiseMultiplicative> : _AnyLayerTangentVectorBox<T.VectorSpaceScalar>
  where T.TangentVector == T, T.VectorSpaceScalar: FloatingPoint & ElementaryFunctions
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

  /// The underlying base value, or `OpaqueScalar` expanded into a full tangent vector.
  var _baseWithScalarUnwrap: T {
    if let scalar = _getOpaqueScalar() {
      return T.zero.adding(scalar)
    } else {
      return _base
    }
  }

  override func _unboxed<U: Differentiable & VectorProtocol & ElementaryFunctions & PointwiseMultiplicative>(as type: U.Type) -> U?
    where U.TangentVector == U, U.VectorSpaceScalar == T.VectorSpaceScalar
  {
    return (self as? _ConcreteAnyLayerTangentVectorBox<U>)?._base
  }

  // `Equatable` requirements (implied by `AdditiveArithmetic`).

  override func _isEqual(to other: _AnyLayerTangentVectorBox<T.VectorSpaceScalar>) -> Bool {
    if let otherScalar = other._getOpaqueScalar() {
      if let scalar = _getOpaqueScalar() {
        return scalar == otherScalar
      } else {
        return _base == T.zero.adding(otherScalar)
      }
    } else if _getOpaqueScalar() != nil {
      return other._isEqual(to: self)
    } else {
      return _base == other._unboxed(as: T.self)
    }
  }

  override func _isNotEqual(to other: _AnyLayerTangentVectorBox<T.VectorSpaceScalar>) -> Bool {
    if let otherScalar = other._getOpaqueScalar() {
      if let scalar = _getOpaqueScalar() {
        return scalar != otherScalar
      } else {
        return _base != T.zero.adding(otherScalar)
      }
    } else if _getOpaqueScalar() != nil {
      return other._isNotEqual(to: self)
    } else {
      return _base != other._unboxed(as: T.self)
    }
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

    guard let xBase = x._unboxed(as: T.self) else {
      _derivativeTypeMismatch(T.self, type(of: x._typeErasedBase))
    }
    return _ConcreteAnyLayerTangentVectorBox(_base + xBase)
  }

  override func _subtract(_ x: _AnyLayerTangentVectorBox<T.VectorSpaceScalar>) -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    // C - x = (1 * C) - x
    if let scalar = _getOpaqueScalar() {
      return _AnyLayerTangentVectorBox<T.VectorSpaceScalar>._one._scaled(by: scalar)._subtract(x)
    }
    // self - C = self - C
    if let scalar = x._getOpaqueScalar() {
      return self._subtracting(scalar)
    }

    guard let xBase = x._unboxed(as: T.self) else {
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
  override class var _one: _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(T.one)
  }

  override func _reciprocal() -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(_base.reciprocal)
  }

  override func _pointwiseMultiply(by: _AnyLayerTangentVectorBox<T.VectorSpaceScalar>) -> _AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return _ConcreteAnyLayerTangentVectorBox<T>(_base .* by._unboxed(as: T.self)!)
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
    return _ConcreteAnyLayerTangentVectorBox<T>(T.pow(_base, y._unboxed(as: T.self)!));
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
        direction._unboxed(as: T.TangentVector.self) else {
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
/// base derivative value conforming to `Differentiable`, `VectorProtocol`,
/// `ElementaryFunctions`, and `PointwiseMultiplicative`, hiding the specifics of the underlying value.
public struct AnyLayerTangentVector<F: FloatingPoint & ElementaryFunctions>: VectorProtocol & KeyPathIterable {
  internal var _box: _AnyLayerTangentVectorBox<F>

  internal init(_box: _AnyLayerTangentVectorBox<F>) {
    self._box = _box
  }

  /// The underlying base value, without logic applied to handle `OpaqueScalar`.
  @usableFromInline
  var _tangentOrScalar: Any {
    return _box._typeErasedBase
  }

  /// The underlying base tangent vector.
  /// This will either be an instance of the underlying layer's tangent vector,
  /// or just a scalar when the tangent vector contains only elements with that value.
  public var base: Any {
    if let scalar = _box._getOpaqueScalar() {
      return scalar
    } else {
      return _box._typeErasedBase
    }
  }

  /// Creates a type-erased wrapper from the given layer.
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
    return (AnyLayerTangentVector<F>(base), { v in v._tangentOrScalar as! T.TangentVector })
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
  public typealias VectorSpaceScalar = F

  // `Equatable` requirements (implied by `AdditiveArithmetic`).
  public static func == (lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector) -> Bool {
    return lhs._box._isEqual(to: rhs._box)
  }
  public static func != (lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector) -> Bool {
    return lhs._box._isNotEqual(to: rhs._box)
  }

  public static func + (
    lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector
  ) -> AnyLayerTangentVector {
    return .init(_box: lhs._box._add(rhs._box))
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
    return .init(_box: lhs._box._subtract(rhs._box))
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

  /// Internal struct representing an opaque scalar value.
  /// This is equivalent to T.TangentVector.zero.adding(scalar)
  /// where T is the actual layer type. Because `zero` and `one` are
  /// static, however, we just capture the scalar value for now and expand
  /// into the actual `TangentVector` type lazily.
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
      return OpaqueScalar(value + x)
    }

    @usableFromInline func subtracting(_ x: F) -> OpaqueScalar {
      return OpaqueScalar(value - x)
    }

    @usableFromInline func scaled(by: F) -> OpaqueScalar {
      return OpaqueScalar(value * by)
    }

    // `PointwiseMultiplicative` requirements.
    @usableFromInline static var one: OpaqueScalar {
      return OpaqueScalar(F(1))
    }

    @usableFromInline var reciprocal: OpaqueScalar {
      return OpaqueScalar(F(1) / value)
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
}

extension AnyLayerTangentVector: Differentiable {
  public mutating func move(along direction: TangentVector) {
    _box._move(along: direction._box)
  }
}

extension AnyLayerTangentVector: EuclideanDifferentiable {
  public var differentiableVectorView: TangentVector {
    return self
  }
}

extension AnyLayerTangentVector: AdditiveArithmetic {
  public static var zero: AnyLayerTangentVector {
    return .init(
      _box: _ConcreteAnyLayerTangentVectorBox<OpaqueScalar>._zero)
  }
  
  public func adding(_ x: VectorSpaceScalar) -> Self {
    return .init(_box: _box._adding(x));
  }

  public func subtracting(_ x: VectorSpaceScalar) -> Self {
    return .init(_box: _box._subtracting(x));
  }

  public func scaled(by scalar: VectorSpaceScalar) -> Self {
    return .init(_box: _box._scaled(by: scalar))
  }
}

extension AnyLayerTangentVector: PointwiseMultiplicative {
  public static var one: AnyLayerTangentVector {
    return .init(_box: _ConcreteAnyLayerTangentVectorBox<OpaqueScalar>._one)
  }

  public var reciprocal: AnyLayerTangentVector {
    return .init(_box: _box._reciprocal())
  }

  public static func .* (lhs: Self, rhs: Self) -> Self {
    return .init(_box: lhs._box._pointwiseMultiply(by: rhs._box))
  }
}

extension AnyLayerTangentVector: ElementaryFunctions {
  public static func sqrt(_ x: Self) -> Self {
    return .init(_box: x._box._sqrt())
  }
  public static func cos(_ x: Self) -> Self {
    return .init(_box: x._box._cos())
  }
  public static func sin(_ x: Self) -> Self {
    return .init(_box: x._box._sin())
  }
  public static func tan(_ x: Self) -> Self {
    return .init(_box: x._box._tan())
  }
  public static func acos(_ x: Self) -> Self {
    return .init(_box: x._box._acos())
  }
  public static func asin(_ x: Self) -> Self {
    return .init(_box: x._box._asin())
  }
  public static func atan(_ x: Self) -> Self {
    return .init(_box: x._box._atan())
  }
  public static func cosh(_ x: Self) -> Self {
    return .init(_box: x._box._cosh())
  }
  public static func sinh(_ x: Self) -> Self {
    return .init(_box: x._box._sinh())
  }
  public static func tanh(_ x: Self) -> Self {
    return .init(_box: x._box._tanh())
  }
  public static func acosh(_ x: Self) -> Self {
    return .init(_box: x._box._acosh())
  }
  public static func asinh(_ x: Self) -> Self {
    return .init(_box: x._box._asinh())
  }
  public static func atanh(_ x: Self) -> Self {
    return .init(_box: x._box._atanh())
  }
  public static func exp(_ x: Self) -> Self {
    return .init(_box: x._box._exp())
  }
  public static func exp2(_ x: Self) -> Self {
    return .init(_box: x._box._exp2())
  }
  public static func exp10(_ x: Self) -> Self {
    return .init(_box: x._box._exp10())
  }
  public static func expm1(_ x: Self) -> Self {
    return .init(_box: x._box._expm1())
  }
  public static func log(_ x: Self) -> Self {
    return .init(_box: x._box._log())
  }
  public static func log2(_ x: Self) -> Self {
    return .init(_box: x._box._log2())
  }
  public static func log10(_ x: Self) -> Self {
    return .init(_box: x._box._log10())
  }
  public static func log1p(_ x: Self) -> Self {
    return .init(_box: x._box._log1p())
  }
  public static func pow(_ x: Self, _ y: Self) -> Self {
    return .init(_box: x._box._pow(y._box))
  }
  public static func pow(_ x: Self, _ n: Int) -> Self {
    return .init(_box: x._box._pow(n))
  }
  public static func root(_ x: Self, _ n: Int) -> Self {
    return .init(_box: x._box._root(n))
  }
}
