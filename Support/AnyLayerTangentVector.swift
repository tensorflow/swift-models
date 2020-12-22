// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
internal class AnyLayerTangentVectorBox {
  /// The underlying value, type-erased to `Any`.
  var typeErasedBase: Any {
    mustOverride()
  }

  /// Returns the underlying value unboxed to the given type, if possible.
  func unboxed<U: TangentVectorConformances>(as type: U.Type) -> U?
  where U.TangentVector == U, U.VectorSpaceScalar == Float {
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
  func _adding(_ x: Float) -> AnyLayerTangentVectorBox {
    mustOverride()
  }
  func _subtracting(_ x: Float) -> AnyLayerTangentVectorBox {
    mustOverride()
  }

  /// Returns `self` multiplied by the given scalar.
  func _scaled(by: Float) -> AnyLayerTangentVectorBox {
    mustOverride()
  }

  // `Differentiable` requirements.
  /// Moves `self` along the given direction. In Riemannian geometry, this is equivalent to exponential map, which moves `self` on the geodesic surface along the given tangent vector.
  func _move(along direction: AnyLayerTangentVector) {
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
  
  /// The expMinusOne function.
  func _expMinusOne() -> AnyLayerTangentVectorBox {
    mustOverride()
  }

  /// The exp10 function.
  func _exp10() -> AnyLayerTangentVectorBox {
    mustOverride()
  }

  /// The exp2 function.
  func _exp2() -> AnyLayerTangentVectorBox {
    mustOverride()
  }

  /// The expm1 function.
  func _expm1() -> AnyLayerTangentVectorBox {
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
  
  /// The log function.
  func _log() -> AnyLayerTangentVectorBox {
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
  func getOpaqueScalar() -> Float? {
    return unboxed(as: AnyLayerTangentVector.OpaqueScalar.self)?.value
  }
}

/// A concrete implementation of the type-erased tangent vector wrapper that forwards to an underlying tangent vector.
internal class ConcreteAnyLayerTangentVectorBox<Underlying: TangentVectorConformances>: AnyLayerTangentVectorBox
where Underlying.TangentVector == Underlying, Underlying.VectorSpaceScalar == Float {
  /// The underlying tangent vector.
  var underlying: Underlying

  init(_ underlying: Underlying) {
    self.underlying = underlying
  }

  /// The underlying tangent vector, type-erased to `Any`.
  override var typeErasedBase: Any {
    return underlying
  }

  override func unboxed<U: TangentVectorConformances>(as type: U.Type) -> U?
  where U.TangentVector == U, U.VectorSpaceScalar == Underlying.VectorSpaceScalar {
    return (self as? ConcreteAnyLayerTangentVectorBox<U>)?.underlying
  }

  override func duplicate() -> AnyLayerTangentVectorBox {
    return ConcreteAnyLayerTangentVectorBox(underlying)
  }

  // `Equatable` requirements (implied by `AdditiveArithmetic`).
  override func _isEqual(to other: AnyLayerTangentVectorBox) -> Bool {
    if let otherScalar = other.getOpaqueScalar() {
      if let scalar = getOpaqueScalar() {
        return scalar == otherScalar
      } else {
        return underlying == Underlying.zero.adding(otherScalar)
      }
    } else if getOpaqueScalar() != nil {
      return other._isEqual(to: self)
    } else {
      return underlying == other.unboxed(as: Underlying.self)
    }
  }

  override func _isNotEqual(to other: AnyLayerTangentVectorBox) -> Bool {
    if let otherScalar = other.getOpaqueScalar() {
      if let scalar = getOpaqueScalar() {
        return scalar != otherScalar
      } else {
        return underlying != Underlying.zero.adding(otherScalar)
      }
    } else if getOpaqueScalar() != nil {
      return other._isNotEqual(to: self)
    } else {
      return underlying != other.unboxed(as: Underlying.self)
    }
  }

  // `AdditiveArithmetic` requirements.
  override class var _zero: AnyLayerTangentVectorBox {
    return ConcreteAnyLayerTangentVectorBox(Underlying.zero)
  }

  override func _add(_ x: AnyLayerTangentVectorBox) -> AnyLayerTangentVectorBox {
    if let scalar = getOpaqueScalar() {
      // use the associative property, self + x = x + self
      return x._adding(scalar)
    }
    
    if let scalar = x.getOpaqueScalar() {
      // add scalar wrapped by `x` to every element of `self`
      return self._adding(scalar)
    }

    guard let xBase = x.unboxed(as: Underlying.self) else {
      derivativeTypeMismatch(got: type(of: x.typeErasedBase), expected: Underlying.self)
    }
    return ConcreteAnyLayerTangentVectorBox(underlying + xBase)
  }

  override func _subtract(_ x: AnyLayerTangentVectorBox) -> AnyLayerTangentVectorBox {
    if let scalar = getOpaqueScalar() {
      // expand by definition of opqaue scalars and perform the original operation
      return AnyLayerTangentVectorBox._one._scaled(by: scalar)._subtract(x)
    }

    if let scalar = x.getOpaqueScalar() {
      // subtract the scalar wrapped by `x` from every element of `self`
      return self._subtracting(scalar)
    }

    guard let xBase = x.unboxed(as: Underlying.self) else {
      derivativeTypeMismatch(got: type(of: x.typeErasedBase), expected: Underlying.self)
    }
    return ConcreteAnyLayerTangentVectorBox(underlying - xBase)
  }
  
  // `VectorProtocol` requirements.
  override func _adding(_ x: Underlying.VectorSpaceScalar) -> AnyLayerTangentVectorBox {
    return ConcreteAnyLayerTangentVectorBox(underlying.adding(x))
  }
  override func _subtracting(_ x: Underlying.VectorSpaceScalar) -> AnyLayerTangentVectorBox {
    return ConcreteAnyLayerTangentVectorBox(underlying.subtracting(x))
  }
  override func _scaled(by: Underlying.VectorSpaceScalar) -> AnyLayerTangentVectorBox {
    return ConcreteAnyLayerTangentVectorBox(underlying.scaled(by: by))
  }

  // `PointwiseMultiplicative` requirements.
  override class var _one: AnyLayerTangentVectorBox {
    return ConcreteAnyLayerTangentVectorBox(Underlying.one)
  }

  override func _reciprocal() -> AnyLayerTangentVectorBox {
    return ConcreteAnyLayerTangentVectorBox(underlying.reciprocal)
  }

  override func _pointwiseMultiply(by: AnyLayerTangentVectorBox) -> AnyLayerTangentVectorBox {
    return ConcreteAnyLayerTangentVectorBox(underlying .* by.unboxed(as: Underlying.self)!)
  }

  // `ElementaryFunctions` requirements.
  override func _sqrt() -> AnyLayerTangentVectorBox {
    return ConcreteAnyLayerTangentVectorBox(Underlying.sqrt(underlying));
  }
  override func _cos() -> AnyLayerTangentVectorBox {
    return ConcreteAnyLayerTangentVectorBox(Underlying.cos(underlying));
  }
  override func _sin() -> AnyLayerTangentVectorBox {
    return ConcreteAnyLayerTangentVectorBox(Underlying.sin(underlying));
  }
  override func _tan() -> AnyLayerTangentVectorBox {
    return ConcreteAnyLayerTangentVectorBox(Underlying.tan(underlying));
  }
  override func _acos() -> AnyLayerTangentVectorBox {
    return ConcreteAnyLayerTangentVectorBox(Underlying.acos(underlying));
  }
  override func _asin() -> AnyLayerTangentVectorBox {
    return ConcreteAnyLayerTangentVectorBox(Underlying.asin(underlying));
  }
  override func _atan() -> AnyLayerTangentVectorBox {
    return ConcreteAnyLayerTangentVectorBox(Underlying.atan(underlying));
  }
  override func _cosh() -> AnyLayerTangentVectorBox {
    return ConcreteAnyLayerTangentVectorBox(Underlying.cosh(underlying));
  }
  override func _sinh() -> AnyLayerTangentVectorBox {
    return ConcreteAnyLayerTangentVectorBox(Underlying.sinh(underlying));
  }
  override func _tanh() -> AnyLayerTangentVectorBox {
    return ConcreteAnyLayerTangentVectorBox(Underlying.tanh(underlying));
  }
  override func _acosh() -> AnyLayerTangentVectorBox {
    return ConcreteAnyLayerTangentVectorBox(Underlying.acosh(underlying));
  }
  override func _asinh() -> AnyLayerTangentVectorBox {
    return ConcreteAnyLayerTangentVectorBox(Underlying.asinh(underlying));
  }
  override func _atanh() -> AnyLayerTangentVectorBox {
    return ConcreteAnyLayerTangentVectorBox(Underlying.atanh(underlying));
  }
  override func _exp() -> AnyLayerTangentVectorBox {
    return ConcreteAnyLayerTangentVectorBox(Underlying.exp(underlying));
  }
  override func _expMinusOne() -> AnyLayerTangentVectorBox {
    // TODO: Re-enable this once we have settled on a single toolchain.
    fatalError("expMinusOne() is currently unimplemented for this toolchain.")
//    return ConcreteAnyLayerTangentVectorBox(Underlying.expMinusOne(underlying));
  }
  override func _log1p() -> AnyLayerTangentVectorBox {
    // TODO: Re-enable this once we have settled on a single toolchain.
    fatalError("log1p() is currently unimplemented for this toolchain.")
    // return ConcreteAnyLayerTangentVectorBox(Underlying.log(onePlus: underlying));
    // return ConcreteAnyLayerTangentVectorBox(Underlying.log1p(underlying));
  }
  override func _exp2() -> AnyLayerTangentVectorBox {
    // TODO: Re-enable this once we have settled on a single toolchain.
    fatalError("exp2() is currently unimplemented for this toolchain.")
    // return ConcreteAnyLayerTangentVectorBox(Underlying.exp2(underlying));
  }
  override func _exp10() -> AnyLayerTangentVectorBox {
    // TODO: Re-enable this once we have settled on a single toolchain.
    fatalError("exp10() is currently unimplemented for this toolchain.")
    // return ConcreteAnyLayerTangentVectorBox(Underlying.exp10(underlying));
  }
  override func _log() -> AnyLayerTangentVectorBox {
    return ConcreteAnyLayerTangentVectorBox(Underlying.log(underlying));
  }
  override func _pow(_ y: AnyLayerTangentVectorBox) -> AnyLayerTangentVectorBox {
    return ConcreteAnyLayerTangentVectorBox(Underlying.pow(underlying, y.unboxed(as: Underlying.self)!));
  }
  override func _pow(_ n: Int) -> AnyLayerTangentVectorBox {
    return ConcreteAnyLayerTangentVectorBox(Underlying.pow(underlying, n));
  }
  override func _root(_ n: Int) -> AnyLayerTangentVectorBox {
    return ConcreteAnyLayerTangentVectorBox(Underlying.root(underlying, n));
  }

  // `Differentiable` requirements.
  override func _move(along direction: AnyLayerTangentVector) {
    if let scalarDirection = direction.box.getOpaqueScalar() {
      underlying.move(along: Underlying.TangentVector.zero.adding(scalarDirection))
    } else {
      guard let directionBase =
        direction.unboxed(as: Underlying.TangentVector.self) else {
        derivativeTypeMismatch(got: type(of: direction.base), expected: Underlying.self)
      }
      underlying.move(along: directionBase)
    }
  }

  // `EuclideanDifferentiable` requirements.
  override var _differentiableVectorView: AnyLayerTangentVectorBox {
    return self
  }
}

/// A type-erased derivative value.
///
/// The `AnyLayerTangentVector` type forwards its operations to an arbitrary underlying
/// base derivative value conforming to `Differentiable`, `VectorProtocol`,
/// `ElementaryFunctions`, and `PointwiseMultiplicative`, hiding the specifics of the underlying value.
public struct AnyLayerTangentVector: KeyPathIterable {
  internal var box: AnyLayerTangentVectorBox

  internal init(box: AnyLayerTangentVectorBox) {
    self.box = box
  }

  /// Returns the underlying value unboxed to the given type, if possible.
  public func unboxed<U: TangentVectorConformances>(as type: U.Type) -> U?
    where U.TangentVector == U, U.VectorSpaceScalar == Float {
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
  public init<Underlying: TangentVectorConformances>(_ underlying: Underlying)
  where Underlying.TangentVector == Underlying, Underlying.VectorSpaceScalar == Float {
    self.box = ConcreteAnyLayerTangentVectorBox(underlying)
  }

  @derivative(of: init)
  @usableFromInline
  internal static func _vjpInit<Underlying: TangentVectorConformances>(
    _ underlying: Underlying
  ) -> (value: AnyLayerTangentVector, pullback: (AnyLayerTangentVector) -> Underlying.TangentVector)
    where Underlying.TangentVector == Underlying, Underlying.VectorSpaceScalar == Float
  {
    return (AnyLayerTangentVector(underlying), { v in v.unboxed(as: Underlying.TangentVector.self)! })
  }

  @derivative(of: init)
  @usableFromInline
  internal static func _jvpInit<Underlying: TangentVectorConformances>(
    _ underlying: Underlying
  ) -> (value: AnyLayerTangentVector, differential: (Underlying.TangentVector) -> AnyLayerTangentVector)
    where Underlying.TangentVector == Underlying, Underlying.VectorSpaceScalar == Float
  {
    return (AnyLayerTangentVector(underlying), { dbase in AnyLayerTangentVector(dbase) })
  }

  public typealias TangentVector = AnyLayerTangentVector

  /// Internal struct representing an opaque scalar value.
  /// This is equivalent to Underlying.TangentVector.zero.adding(scalar)
  /// where T is the actual layer type. Because `zero` and `one` are
  /// static, however, we just capture the scalar value for now and expand
  /// into the actual `TangentVector` type lazily.
  @frozen
  @usableFromInline
  internal struct OpaqueScalar: TangentVectorConformances {
    @usableFromInline typealias VectorSpaceScalar = Float
    let value: Float

    @usableFromInline typealias TangentVector = OpaqueScalar

    init(_ value: Float) {
      self.value = value
    }

    // `VectorProtocol` requirements.
    @usableFromInline func adding(_ x: Float) -> OpaqueScalar {
      return OpaqueScalar(value + x)
    }

    @usableFromInline func subtracting(_ x: Float) -> OpaqueScalar {
      return OpaqueScalar(value - x)
    }

    @usableFromInline func scaled(by: Float) -> OpaqueScalar {
      return OpaqueScalar(value * by)
    }

    // `PointwiseMultiplicative` requirements.
    @usableFromInline static var one: OpaqueScalar {
      return OpaqueScalar(Float(1))
    }

    @usableFromInline var reciprocal: OpaqueScalar {
      return OpaqueScalar(Float(1) / value)
    }

    @usableFromInline static func .* (lhs: OpaqueScalar, rhs: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(lhs.value * rhs.value)
    }

    // `ElementaryFunctions` requirements.
    @usableFromInline static func sqrt(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(Float.sqrt(x.value))
    }

    @usableFromInline static func cos(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(Float.cos(x.value))
    }

    @usableFromInline static func sin(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(Float.sin(x.value))
    }

    @usableFromInline static func tan(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(Float.tan(x.value))
    }

    @usableFromInline static func acos(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(Float.acos(x.value))
    }

    @usableFromInline static func asin(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(Float.asin(x.value))
    }

    @usableFromInline static func atan(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(Float.atan(x.value))
    }

    @usableFromInline static func cosh(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(Float.cosh(x.value))
    }

    @usableFromInline static func sinh(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(Float.sinh(x.value))
    }

    @usableFromInline static func tanh(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(Float.tanh(x.value))
    }

    @usableFromInline static func acosh(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(Float.acosh(x.value))
    }

    @usableFromInline static func asinh(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(Float.asinh(x.value))
    }

    @usableFromInline static func atanh(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(Float.atanh(x.value))
    }

    @usableFromInline static func exp(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(Float.exp(x.value))
    }

    @usableFromInline static func expMinusOne(_ x: OpaqueScalar) -> OpaqueScalar {
      // TODO: Re-enable this once we have settled on a single toolchain.
      fatalError("expMinusOne() is currently unimplemented for this toolchain.")
      // return OpaqueScalar(Float.expMinusOne(x.value))
    }

    @usableFromInline static func log(onePlus x: OpaqueScalar) -> OpaqueScalar {
      // TODO: Re-enable this once we have settled on a single toolchain.
      fatalError("log(onePlus:) is currently unimplemented for this toolchain.")
      // return OpaqueScalar(Float.log(onePlus: x.value))
    }

    @usableFromInline static func log(_ x: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(Float.log(x.value))
    }

    @usableFromInline static func pow(_ x: OpaqueScalar, _ y: OpaqueScalar) -> OpaqueScalar {
      return OpaqueScalar(Float.pow(x.value, y.value))
    }

    @usableFromInline static func pow(_ x: OpaqueScalar, _ n: Int) -> OpaqueScalar {
      return OpaqueScalar(Float.pow(x.value, n))
    }

    @usableFromInline static func root(_ x: OpaqueScalar, _ n: Int) -> OpaqueScalar {
      return OpaqueScalar(Float.root(x.value, n))
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
  public typealias VectorSpaceScalar = Float

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
  public static func expMinusOne(_ x: Self) -> Self {
    return .init(box: x.box._expMinusOne())
  }
  public static func log(onePlus x: Self) -> Self {
    return .init(box: x.box._log1p())
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
  public static func log2(_ x: Self) -> Self {
    return .init(box: x.box._log2())
  }
  public static func log10(_ x: Self) -> Self {
    return .init(box: x.box._log10())
  }
  public static func log1p(_ x: Self) -> Self {
    return .init(box: x.box._log1p())
  }
  public static func log(_ x: Self) -> Self {
    return .init(box: x.box._log())
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
