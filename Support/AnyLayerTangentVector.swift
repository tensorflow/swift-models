import TensorFlow
import _Differentiation

/// The base type for a type-erased box that encapsulates a layer's tangent vector.
/// Offers forwarders to implement conformance to `Equatable`, `AdditiveArithmetic`, `Differentiable`,
/// `EuclideanDifferentiable`, `PointwiseMultiplicative`, and `ElementaryFunctions`.
internal class AnyLayerTangentVectorBox<F: FloatingPoint & ElementaryFunctions> {
  /// The underlying base value, type-erased to `Any`.
  var typeErasedBase: Any {
    fatalError("Must implement")
  }

  /// Returns the underlying value unboxed to the given type, if possible.
  func unboxed<U: Differentiable & VectorProtocol & ElementaryFunctions & PointwiseMultiplicative>(as type: U.Type) -> U?
    where U.TangentVector == U, U.VectorSpaceScalar == F {
    fatalError("Must implement")
  }
  
  // `Equatable` requirements (implied by `AdditiveArithmetic`).
  func _isEqual(to other: AnyLayerTangentVectorBox) -> Bool {
    fatalError("Must implement")
  }
  func _isNotEqual(to other: AnyLayerTangentVectorBox) -> Bool {
    fatalError("Must implement")
  }

  func _add(_ x: AnyLayerTangentVectorBox) -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _subtract(_ x: AnyLayerTangentVectorBox) -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }

  // `AdditiveArithmetic` requirements.
  class var _zero: AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  
  func _adding(_ x: F) -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _subtracting(_ x: F) -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _scaled(by: F) -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }

  // `Differentiable` requirements.
  func _move(along direction: AnyLayerTangentVectorBox) {
    fatalError("Must implement")
  }

  // `EuclideanDifferentiable` requirements.
  var _differentiableVectorView: AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }

  // `PointwiseMultiplicative` requirements.
  class var _one: AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  
  func _reciprocal() -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }

  func _pointwiseMultiply(by: AnyLayerTangentVectorBox) -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }

  // TODO(shadaj): split out into separate type?
  // `ElementaryFunctions` requirements.
  func _sqrt() -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _cos() -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _sin() -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _tan() -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _acos() -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _asin() -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _atan() -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _cosh() -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _sinh() -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _tanh() -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _acosh() -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _asinh() -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _atanh() -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _exp() -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _exp2() -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _exp10() -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _expm1() -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _log() -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _log2() -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _log10() -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _log1p() -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _pow(_ y: AnyLayerTangentVectorBox) -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _pow(_ n: Int) -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
  func _root(_ n: Int) -> AnyLayerTangentVectorBox {
    fatalError("Must implement")
  }
}

extension AnyLayerTangentVectorBox {
  /// Optionally returns the underlying scalar if the wrapped value has type `AnyLayerTangentVector.OpaqueScalar`.
  func getOpaqueScalar() -> F? {
    return unboxed(as: AnyLayerTangentVector<F>.OpaqueScalar.self)?.value
  }
}

/// A concrete implementation of the type-erased tangent vector wrapper that forwards to an underlying tangent vector.
internal class ConcreteAnyLayerTangentVectorBox<T: Differentiable & VectorProtocol & ElementaryFunctions & PointwiseMultiplicative> : AnyLayerTangentVectorBox<T.VectorSpaceScalar>
  where T.TangentVector == T, T.VectorSpaceScalar: FloatingPoint & ElementaryFunctions
{
  /// The underlying base value.
  var base: T

  init(_ base: T) {
    self.base = base
  }

  /// The underlying base value, type-erased to `Any`.
  override var typeErasedBase: Any {
    return base
  }

  override func unboxed<U: Differentiable & VectorProtocol & ElementaryFunctions & PointwiseMultiplicative>(as type: U.Type) -> U?
    where U.TangentVector == U, U.VectorSpaceScalar == T.VectorSpaceScalar
  {
    return (self as? ConcreteAnyLayerTangentVectorBox<U>)?.base
  }

  // `Equatable` requirements (implied by `AdditiveArithmetic`).
  override func _isEqual(to other: AnyLayerTangentVectorBox<T.VectorSpaceScalar>) -> Bool {
    if let otherScalar = other.getOpaqueScalar() {
      if let scalar = getOpaqueScalar() {
        return scalar == otherScalar
      } else {
        return base == T.zero.adding(otherScalar)
      }
    } else if getOpaqueScalar() != nil {
      return other._isEqual(to: self)
    } else {
      return base == other.unboxed(as: T.self)
    }
  }

  override func _isNotEqual(to other: AnyLayerTangentVectorBox<T.VectorSpaceScalar>) -> Bool {
    if let otherScalar = other.getOpaqueScalar() {
      if let scalar = getOpaqueScalar() {
        return scalar != otherScalar
      } else {
        return base != T.zero.adding(otherScalar)
      }
    } else if getOpaqueScalar() != nil {
      return other._isNotEqual(to: self)
    } else {
      return base != other.unboxed(as: T.self)
    }
  }

  override func _add(_ x: AnyLayerTangentVectorBox<T.VectorSpaceScalar>) -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    // C + x = x + C
    if let scalar = getOpaqueScalar() {
      return x._adding(scalar)
    }
    // self + C = self + C
    if let scalar = x.getOpaqueScalar() {
      return self._adding(scalar)
    }

    guard let xBase = x.unboxed(as: T.self) else {
      derivativeTypeMismatch(T.self, type(of: x.typeErasedBase))
    }
    return ConcreteAnyLayerTangentVectorBox(base + xBase)
  }

  override func _subtract(_ x: AnyLayerTangentVectorBox<T.VectorSpaceScalar>) -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    // C - x = (1 * C) - x
    if let scalar = getOpaqueScalar() {
      return AnyLayerTangentVectorBox<T.VectorSpaceScalar>._one._scaled(by: scalar)._subtract(x)
    }
    // self - C = self - C
    if let scalar = x.getOpaqueScalar() {
      return self._subtracting(scalar)
    }

    guard let xBase = x.unboxed(as: T.self) else {
      derivativeTypeMismatch(T.self, type(of: x.typeErasedBase))
    }
    return ConcreteAnyLayerTangentVectorBox(base - xBase)
  }

  // `AdditiveArithmetic` requirements.
  override class var _zero: AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox(T.zero)
  }
  
  override func _adding(_ x: T.VectorSpaceScalar) -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(base.adding(x))
  }
  override func _subtracting(_ x: T.VectorSpaceScalar) -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(base.subtracting(x))
  }
  override func _scaled(by: T.VectorSpaceScalar) -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(base.scaled(by: by))
  }

  // `PointwiseMultiplicative` requirements.
  override class var _one: AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(T.one)
  }

  override func _reciprocal() -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(base.reciprocal)
  }

  override func _pointwiseMultiply(by: AnyLayerTangentVectorBox<T.VectorSpaceScalar>) -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(base .* by.unboxed(as: T.self)!)
  }

  // `ElementaryFunctions` requirements.
  override func _sqrt() -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(T.sqrt(base));
  }
  override func _cos() -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(T.cos(base));
  }
  override func _sin() -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(T.sin(base));
  }
  override func _tan() -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(T.tan(base));
  }
  override func _acos() -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(T.acos(base));
  }
  override func _asin() -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(T.asin(base));
  }
  override func _atan() -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(T.atan(base));
  }
  override func _cosh() -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(T.cosh(base));
  }
  override func _sinh() -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(T.sinh(base));
  }
  override func _tanh() -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(T.tanh(base));
  }
  override func _acosh() -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(T.acosh(base));
  }
  override func _asinh() -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(T.asinh(base));
  }
  override func _atanh() -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(T.atanh(base));
  }
  override func _exp() -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(T.exp(base));
  }
  override func _exp2() -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(T.exp2(base));
  }
  override func _exp10() -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(T.exp10(base));
  }
  override func _expm1() -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(T.expm1(base));
  }
  override func _log() -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(T.log(base));
  }
  override func _log2() -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(T.log2(base));
  }
  override func _log10() -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(T.log10(base));
  }
  override func _log1p() -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(T.log1p(base));
  }
  override func _pow(_ y: AnyLayerTangentVectorBox<T.VectorSpaceScalar>) -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(T.pow(base, y.unboxed(as: T.self)!));
  }
  override func _pow(_ n: Int) -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(T.pow(base, n));
  }
  override func _root(_ n: Int) -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<T>(T.root(base, n));
  }

  // `Differentiable` requirements.
  override func _move(along direction: AnyLayerTangentVectorBox<T.VectorSpaceScalar>) {
    if let scalarDirection = direction.getOpaqueScalar() {
      base.move(along: T.TangentVector.zero.adding(scalarDirection))
    } else {
      guard let directionBase =
        direction.unboxed(as: T.TangentVector.self) else {
        derivativeTypeMismatch(T.self, type(of: direction.typeErasedBase))
      }
      base.move(along: directionBase)
    }
  }

  // `EuclideanDifferentiable` requirements.
  override var _differentiableVectorView: AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return self
  }
}

/// A type-erased derivative value.
///
/// The `AnyLayerTangentVector` type forwards its operations to an arbitrary underlying
/// base derivative value conforming to `Differentiable`, `VectorProtocol`,
/// `ElementaryFunctions`, and `PointwiseMultiplicative`, hiding the specifics of the underlying value.
public struct AnyLayerTangentVector<F: FloatingPoint & ElementaryFunctions>: VectorProtocol & KeyPathIterable {
  internal var box: AnyLayerTangentVectorBox<F>

  internal init(box: AnyLayerTangentVectorBox<F>) {
    self.box = box
  }

  /// Returns the underlying value unboxed to the given type, if possible.
  public func unboxed<U: Differentiable & VectorProtocol & ElementaryFunctions & PointwiseMultiplicative>(as type: U.Type) -> U?
    where U.TangentVector == U, U.VectorSpaceScalar == F {
    return box.unboxed(as: type)
  }

  /// The underlying base tangent vector.
  /// This will either be an instance of the underlying layer's tangent vector,
  /// or just a scalar when the tangent vector contains only elements with that value.
  public var base: Any {
    if let scalar = box.getOpaqueScalar() {
      return scalar
    } else {
      return box.typeErasedBase
    }
  }

  /// Creates a type-erased wrapper from the given layer.
  @differentiable
  public init<T: Differentiable & VectorProtocol & ElementaryFunctions & PointwiseMultiplicative>(_ base: T) where T.TangentVector == T, T.VectorSpaceScalar == F {
    self.box = ConcreteAnyLayerTangentVectorBox<T>(base)
  }

  @derivative(of: init)
  @usableFromInline
  internal static func _vjpInit<T: Differentiable & VectorProtocol & ElementaryFunctions & PointwiseMultiplicative>(
    _ base: T
  ) -> (value: AnyLayerTangentVector<F>, pullback: (AnyLayerTangentVector<F>) -> T.TangentVector)
    where T.TangentVector == T, T.VectorSpaceScalar == F
  {
    return (AnyLayerTangentVector<F>(base), { v in v.unboxed(as: T.TangentVector.self)! })
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
    return lhs.box._isEqual(to: rhs.box)
  }
  public static func != (lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector) -> Bool {
    return lhs.box._isNotEqual(to: rhs.box)
  }

  public static func + (
    lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector
  ) -> AnyLayerTangentVector {
    return .init(box: lhs.box._add(rhs.box))
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
    return .init(box: lhs.box._subtract(rhs.box))
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
    box._move(along: direction.box)
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
      box: ConcreteAnyLayerTangentVectorBox<OpaqueScalar>._zero)
  }
  
  public func adding(_ x: VectorSpaceScalar) -> Self {
    return .init(box: box._adding(x));
  }

  public func subtracting(_ x: VectorSpaceScalar) -> Self {
    return .init(box: box._subtracting(x));
  }

  public func scaled(by scalar: VectorSpaceScalar) -> Self {
    return .init(box: box._scaled(by: scalar))
  }
}

extension AnyLayerTangentVector: PointwiseMultiplicative {
  public static var one: AnyLayerTangentVector {
    return .init(box: ConcreteAnyLayerTangentVectorBox<OpaqueScalar>._one)
  }

  public var reciprocal: AnyLayerTangentVector {
    return .init(box: box._reciprocal())
  }

  public static func .* (lhs: Self, rhs: Self) -> Self {
    return .init(box: lhs.box._pointwiseMultiply(by: rhs.box))
  }
}

extension AnyLayerTangentVector: ElementaryFunctions {
  public static func sqrt(_ x: Self) -> Self {
    return .init(box: x.box._sqrt())
  }
  public static func cos(_ x: Self) -> Self {
    return .init(box: x.box._cos())
  }
  public static func sin(_ x: Self) -> Self {
    return .init(box: x.box._sin())
  }
  public static func tan(_ x: Self) -> Self {
    return .init(box: x.box._tan())
  }
  public static func acos(_ x: Self) -> Self {
    return .init(box: x.box._acos())
  }
  public static func asin(_ x: Self) -> Self {
    return .init(box: x.box._asin())
  }
  public static func atan(_ x: Self) -> Self {
    return .init(box: x.box._atan())
  }
  public static func cosh(_ x: Self) -> Self {
    return .init(box: x.box._cosh())
  }
  public static func sinh(_ x: Self) -> Self {
    return .init(box: x.box._sinh())
  }
  public static func tanh(_ x: Self) -> Self {
    return .init(box: x.box._tanh())
  }
  public static func acosh(_ x: Self) -> Self {
    return .init(box: x.box._acosh())
  }
  public static func asinh(_ x: Self) -> Self {
    return .init(box: x.box._asinh())
  }
  public static func atanh(_ x: Self) -> Self {
    return .init(box: x.box._atanh())
  }
  public static func exp(_ x: Self) -> Self {
    return .init(box: x.box._exp())
  }
  public static func exp2(_ x: Self) -> Self {
    return .init(box: x.box._exp2())
  }
  public static func exp10(_ x: Self) -> Self {
    return .init(box: x.box._exp10())
  }
  public static func expm1(_ x: Self) -> Self {
    return .init(box: x.box._expm1())
  }
  public static func log(_ x: Self) -> Self {
    return .init(box: x.box._log())
  }
  public static func log2(_ x: Self) -> Self {
    return .init(box: x.box._log2())
  }
  public static func log10(_ x: Self) -> Self {
    return .init(box: x.box._log10())
  }
  public static func log1p(_ x: Self) -> Self {
    return .init(box: x.box._log1p())
  }
  public static func pow(_ x: Self, _ y: Self) -> Self {
    return .init(box: x.box._pow(y.box))
  }
  public static func pow(_ x: Self, _ n: Int) -> Self {
    return .init(box: x.box._pow(n))
  }
  public static func root(_ x: Self, _ n: Int) -> Self {
    return .init(box: x.box._root(n))
  }
}
