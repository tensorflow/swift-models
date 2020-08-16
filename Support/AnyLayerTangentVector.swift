import TensorFlow
import _Differentiation

fileprivate func mustOverride(function: StaticString = #function, file: StaticString = #file, line: UInt = #line) -> Never {
  fatalError("Function AnyLayerTangentVectorBox.\(function) (defined at: \(file):\(line)) must be overridden.")
}

/// The set of protocol conformances required for the `TangentVector` of a `Layer`.
public typealias TangentVectorConformances = Differentiable & VectorProtocol & ElementaryFunctions & PointwiseMultiplicative

/// The base type for a type-erased box that encapsulates a layer's tangent vector.
/// Offers forwarders to implement conformance to `Equatable`, `AdditiveArithmetic`, `Differentiable`,
/// `EuclideanDifferentiable`, `PointwiseMultiplicative`, and `ElementaryFunctions`.
/// Type Parameters:
///   - Scalar: the scalar type of the underlying tangent vector
internal class AnyLayerTangentVectorBox<Scalar: FloatingPoint & ElementaryFunctions> {
  /// The underlying value, type-erased to `Any`.
  var typeErasedBase: Any {
    mustOverride()
  }

  /// Returns the underlying value unboxed to the given type, if possible.
  func unboxed<U: TangentVectorConformances>(as type: U.Type) -> U?
  where U.TangentVector == U, U.VectorSpaceScalar == Scalar {
    mustOverride()
  }

  // Creates a new box storing a copy of the underlying tangent vector, used to preserve value semantics.
  func duplicate() -> AnyLayerTangentVectorBox {
    mustOverride()
  }
  
  // `Equatable` requirements (implied by `AdditiveArithmetic`).
  /// Returns a Boolean value indicating whether two values are equal.
  func _isEqual(to other: AnyLayerTangentVectorBox) -> Bool {
    mustOverride()
  }

  /// Returns a Boolean value indicating whether two values are not equal.
  func _isNotEqual(to other: AnyLayerTangentVectorBox) -> Bool {
    mustOverride()
  }

  // `AdditiveArithmetic` requirements.
  /// The zero value.
  class var _zero: AnyLayerTangentVectorBox {
    mustOverride()
  }

  /// Adds two values and produces their sum.
  func _add(_ x: AnyLayerTangentVectorBox) -> AnyLayerTangentVectorBox {
    mustOverride()
  }

  /// Subtracts one value from another and produces their difference.
  func _subtract(_ x: AnyLayerTangentVectorBox) -> AnyLayerTangentVectorBox {
    mustOverride()
  }
  
  // `VectorProtocol` requirements.
  func _adding(_ x: Scalar) -> AnyLayerTangentVectorBox {
    mustOverride()
  }
  func _subtracting(_ x: Scalar) -> AnyLayerTangentVectorBox {
    mustOverride()
  }

  /// Returns `self` multiplied by the given scalar.
  func _scaled(by: Scalar) -> AnyLayerTangentVectorBox {
    mustOverride()
  }

  // `Differentiable` requirements.
  /// Moves `self` along the given direction. In Riemannian geometry, this is equivalent to exponential map, which moves `self` on the geodesic surface along the given tangent vector.
  func _move(along direction: AnyLayerTangentVector<Scalar>) {
    mustOverride()
  }

  // `EuclideanDifferentiable` requirements.
  /// The differentiable vector component of `self`.
  var _differentiableVectorView: AnyLayerTangentVectorBox {
    mustOverride()
  }

  // `PointwiseMultiplicative` requirements.
  /// The one value.
  /// One is the identity element for multiplication. For any value, `x .* .one == x` and `.one .* x == x`.
  class var _one: AnyLayerTangentVectorBox {
    mustOverride()
  }
  
  /// The multiplicative inverse of self.
  /// For any value, `x .* x.reciprocal == .one` and `x.reciprocal .* x == .one`.
  func _reciprocal() -> AnyLayerTangentVectorBox {
    mustOverride()
  }

  /// Multiplies two values and produces their product.
  func _pointwiseMultiply(by: AnyLayerTangentVectorBox) -> AnyLayerTangentVectorBox {
    mustOverride()
  }

  // `ElementaryFunctions` requirements.
  /// The square root of `x`.
  /// For real types, if the argument is negative, either the result is NaN or a precondition failure occurs. For complex types, this function has a branch cut along the negative real axis.
  func _sqrt() -> AnyLayerTangentVectorBox {
    mustOverride()
  }
  
  /// The cosine of `x`.
  /// For real types, `x` is interpreted as an angle measured in radians.
  func _cos() -> AnyLayerTangentVectorBox {
    mustOverride()
  }
  
  /// The sine of `x`.
  /// For real types, `x` is interpreted as an angle measured in radians.
  func _sin() -> AnyLayerTangentVectorBox {
    mustOverride()
  }
  
  /// The tangent of `x`.
  func _tan() -> AnyLayerTangentVectorBox {
    mustOverride()
  }
  
  /// The acos function.
  func _acos() -> AnyLayerTangentVectorBox {
    mustOverride()
  }
  
  /// The asin function.
  func _asin() -> AnyLayerTangentVectorBox {
    mustOverride()
  }
 
  /// The atan function.
  func _atan() -> AnyLayerTangentVectorBox {
    mustOverride()
  }
  
  /// The cosh function.
  func _cosh() -> AnyLayerTangentVectorBox {
    mustOverride()
  }
  
  /// The sinh function.
  func _sinh() -> AnyLayerTangentVectorBox {
    mustOverride()
  }
  
  /// The tanh function.
  func _tanh() -> AnyLayerTangentVectorBox {
    mustOverride()
  }
  
  /// The acosh function.
  func _acosh() -> AnyLayerTangentVectorBox {
    mustOverride()
  }
  
  /// The asinh function.
  func _asinh() -> AnyLayerTangentVectorBox {
    mustOverride()
  }
  
  /// The atanh function.
  func _atanh() -> AnyLayerTangentVectorBox {
    mustOverride()
  }
  
  /// The exp function.
  func _exp() -> AnyLayerTangentVectorBox {
    mustOverride()
  }
  
  /// The exp2 function.
  func _exp2() -> AnyLayerTangentVectorBox {
    mustOverride()
  }
  
  /// The exp10 function.
  func _exp10() -> AnyLayerTangentVectorBox {
    mustOverride()
  }
  
  /// The expm1 function.
  func _expm1() -> AnyLayerTangentVectorBox {
    mustOverride()
  }
  
  /// The log function.
  func _log() -> AnyLayerTangentVectorBox {
    mustOverride()
  }
  
  /// The log2 function.
  func _log2() -> AnyLayerTangentVectorBox {
    mustOverride()
  }
  
  /// The log10 function.
  func _log10() -> AnyLayerTangentVectorBox {
    mustOverride()
  }
  
  /// The log1p function.
  func _log1p() -> AnyLayerTangentVectorBox {
    mustOverride()
  }
  
  /// `exp(y log(x))` computed without loss of intermediate precision.
  /// For real types, if `x` is negative the result is NaN, even if `y` has an integral value. For complex types, there is a branch cut on the negative real axis.
  func _pow(_ y: AnyLayerTangentVectorBox) -> AnyLayerTangentVectorBox {
    mustOverride()
  }
  
  /// `x` raised to the `n`th power.
  func _pow(_ n: Int) -> AnyLayerTangentVectorBox {
    mustOverride()
  }
  
  /// The `n`th root of `x`.
  /// For real types, if `x` is negative and `n` is even, the result is NaN. For complex types, there is a branch cut along the negative real axis.
  func _root(_ n: Int) -> AnyLayerTangentVectorBox {
    mustOverride()
  }
}

extension AnyLayerTangentVectorBox {
  /// Optionally returns the underlying scalar if the wrapped value has type `AnyLayerTangentVector.OpaqueScalar`.
  func getOpaqueScalar() -> Scalar? {
    return unboxed(as: AnyLayerTangentVector<Scalar>.OpaqueScalar.self)?.value
  }
}

/// A concrete implementation of the type-erased tangent vector wrapper that forwards to an underlying tangent vector.
internal class ConcreteAnyLayerTangentVectorBox<T: TangentVectorConformances>: AnyLayerTangentVectorBox<T.VectorSpaceScalar>
where T.TangentVector == T, T.VectorSpaceScalar: FloatingPoint & ElementaryFunctions {
  /// The underlying base value.
  var base: T

  init(_ base: T) {
    self.base = base
  }

  /// The underlying base value, type-erased to `Any`.
  override var typeErasedBase: Any {
    return base
  }

  override func unboxed<U: TangentVectorConformances>(as type: U.Type) -> U?
  where U.TangentVector == U, U.VectorSpaceScalar == T.VectorSpaceScalar {
    return (self as? ConcreteAnyLayerTangentVectorBox<U>)?.base
  }

  override func duplicate() -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox(base)
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

  // `AdditiveArithmetic` requirements.
  override class var _zero: AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox(T.zero)
  }

  override func _add(_ x: AnyLayerTangentVectorBox<T.VectorSpaceScalar>) -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    if let scalar = getOpaqueScalar() {
      // use the associative property, self + x = x + self
      return x._adding(scalar)
    }
    
    if let scalar = x.getOpaqueScalar() {
      // add scalar wrapped by `x` to every element of `self`
      return self._adding(scalar)
    }

    guard let xBase = x.unboxed(as: T.self) else {
      derivativeTypeMismatch(got: type(of: x.typeErasedBase), expected: T.self)
    }
    return ConcreteAnyLayerTangentVectorBox(base + xBase)
  }

  override func _subtract(_ x: AnyLayerTangentVectorBox<T.VectorSpaceScalar>) -> AnyLayerTangentVectorBox<T.VectorSpaceScalar> {
    if let scalar = getOpaqueScalar() {
      // expand by definition of opqaue scalars and perform the original operation
      return AnyLayerTangentVectorBox<T.VectorSpaceScalar>._one._scaled(by: scalar)._subtract(x)
    }

    if let scalar = x.getOpaqueScalar() {
      // subtract the scalar wrapped by `x` from every element of `self`
      return self._subtracting(scalar)
    }

    guard let xBase = x.unboxed(as: T.self) else {
      derivativeTypeMismatch(got: type(of: x.typeErasedBase), expected: T.self)
    }
    return ConcreteAnyLayerTangentVectorBox(base - xBase)
  }
  
  // `VectorProtocol` requirements.
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
  override func _move(along direction: AnyLayerTangentVector<T.VectorSpaceScalar>) {
    if let scalarDirection = direction.box.getOpaqueScalar() {
      base.move(along: T.TangentVector.zero.adding(scalarDirection))
    } else {
      guard let directionBase =
        direction.unboxed(as: T.TangentVector.self) else {
        derivativeTypeMismatch(got: type(of: direction.base), expected: T.self)
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
public struct AnyLayerTangentVector<F: FloatingPoint & ElementaryFunctions>: KeyPathIterable {
  internal var box: AnyLayerTangentVectorBox<F>

  internal init(box: AnyLayerTangentVectorBox<F>) {
    self.box = box
  }

  /// Returns the underlying value unboxed to the given type, if possible.
  public func unboxed<U: TangentVectorConformances>(as type: U.Type) -> U?
    where U.TangentVector == U, U.VectorSpaceScalar == F {
    return box.unboxed(as: type)
  }

  /// The underlying base tangent vector.
  /// This will either be an instance of the underlying layer's tangent vector type,
  /// or just a scalar when the tangent vector contains only elements with that value.
  public var base: Any {
    if let scalar = box.getOpaqueScalar() {
      return scalar
    } else {
      return box.typeErasedBase
    }
  }

  /// Creates a type-erased wrapper from the given tangent vector.
  @differentiable
  public init<T: TangentVectorConformances>(_ base: T)
  where T.TangentVector == T, T.VectorSpaceScalar == F {
    self.box = ConcreteAnyLayerTangentVectorBox<T>(base)
  }

  @derivative(of: init)
  @usableFromInline
  internal static func _vjpInit<T: TangentVectorConformances>(
    _ base: T
  ) -> (value: AnyLayerTangentVector<F>, pullback: (AnyLayerTangentVector<F>) -> T.TangentVector)
    where T.TangentVector == T, T.VectorSpaceScalar == F
  {
    return (AnyLayerTangentVector<F>(base), { v in v.unboxed(as: T.TangentVector.self)! })
  }

  @derivative(of: init)
  @usableFromInline
  internal static func _jvpInit<T: TangentVectorConformances>(
    _ base: T
  ) -> (value: AnyLayerTangentVector<F>, differential: (T.TangentVector) -> AnyLayerTangentVector<F>)
    where T.TangentVector == T, T.VectorSpaceScalar == F
  {
    return (AnyLayerTangentVector<F>(base), { dbase in AnyLayerTangentVector<F>(dbase) })
  }

  public typealias TangentVector = AnyLayerTangentVector

  /// Internal struct representing an opaque scalar value.
  /// This is equivalent to T.TangentVector.zero.adding(scalar)
  /// where T is the actual layer type. Because `zero` and `one` are
  /// static, however, we just capture the scalar value for now and expand
  /// into the actual `TangentVector` type lazily.
  @frozen
  @usableFromInline
  internal struct OpaqueScalar: TangentVectorConformances {
    @usableFromInline typealias VectorSpaceScalar = F
    let value: F

    @usableFromInline typealias TangentVector = OpaqueScalar

    init(_ value: F) {
      self.value = value
    }

    // `VectorProtocol` requirements.
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

extension AnyLayerTangentVector: Equatable {
  public static func == (lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector) -> Bool {
    return lhs.box._isEqual(to: rhs.box)
  }
  
  public static func != (lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector) -> Bool {
    return lhs.box._isNotEqual(to: rhs.box)
  }
}

extension AnyLayerTangentVector: Differentiable {
  public mutating func move(along direction: TangentVector) {
    if !isKnownUniquelyReferenced(&box) { // preserve value semantics
      self.box = box.duplicate()
    }

    box._move(along: direction)
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
}

extension AnyLayerTangentVector: VectorProtocol {
  public typealias VectorSpaceScalar = F

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
