import TensorFlow
import _Differentiation

/// A helper that stops the program with an error when an erased derivative type does not
/// match up with the true underlying type.
@inline(never)
@usableFromInline
internal func derivativeTypeMismatch(
  got: Any.Type, expected: Any.Type, file: StaticString = #file, line: UInt = #line
) -> Never {
  preconditionFailure("""
    Derivative type mismatch: \
    got \(String(reflecting: got)) but expected \(String(reflecting: expected))
    """, file: file, line: line)
}

fileprivate func mustOverride(function: StaticString = #function, file: StaticString = #file, line: UInt = #line) -> Never {
  fatalError("Function AnyLayerBox.\(function) (defined at: \(file):\(line)) must be overridden.")
}

/// The base type for a type-erased box that encapsulates a layer.
/// Offers forwarders to implement conformance to `Layer` and `CopyableToDevice`.
///
/// Type Parameters:
///   - Input: the input type of the underlying layar
///   - Output: the output type of the underlying layer
///   - Scalar: the scalar type of the underlying tangent vector
internal class AnyLayerBox<Input: Differentiable, Output: Differentiable, Scalar: FloatingPoint & ElementaryFunctions> {
  /// The underlying layer, type-erased to `Any`.
  var typeErasedBase: Any {
    mustOverride()
  }

  /// Returns the underlying layer unboxed to the given type, if possible.
  func unboxed<U: Layer>(to type: U.Type) -> U?
  where U.TangentVector.VectorSpaceScalar == Scalar {
    mustOverride()
  }
  
  // `Differentiable` requirements.
  /// Moves `self` along the given direction. In Riemannian geometry, this is equivalent to exponential map, which moves `self` on the geodesic surface along the given tangent vector.
  func _move(along direction: AnyLayerTangentVector<Scalar>) {
    mustOverride()
  }

  // `EuclideanDifferentiable` requirements.
  /// The differentiable vector component of `self`.
  var _differentiableVectorView: AnyLayerTangentVector<Scalar> {
    mustOverride()
  }

  // `Layer` requirements.
  /// Returns the output obtained from applying the layer to the given input.
  func _callAsFunction(_ input: Input) -> Output {
    mustOverride()
  }

  func _vjpCallAsFunction(_ input: Input) ->
    (value: Output, pullback: (Output.TangentVector) -> (AnyLayerTangentVector<Scalar>, Input.TangentVector)) {
    mustOverride()
  }

  // `CopyableToDevice` requirements.
  /// Creates a copy of `self` on the given Device.
  /// All cross-device references are moved to the given Device.
  func _copyToDevice(to device: Device) -> AnyLayerBox {
    mustOverride()
  }

  /// Creates a new box storing a copy of the underlying layer, used to preserve value semantics.
  func duplicate() -> AnyLayerBox<Input, Output, Scalar> {
    mustOverride()
  }
}

/// A concrete implementation of the type-erased layer wrapper that forwards to an underlying layer.
internal class ConcreteLayerBox<Underlying: Layer>: AnyLayerBox<Underlying.Input, Underlying.Output, Underlying.TangentVector.VectorSpaceScalar>
where Underlying.TangentVector.VectorSpaceScalar: FloatingPoint & ElementaryFunctions {
  /// The underlying layer.
  var underlying: Underlying

  /// Constructs the type-erased wrapper given the underlying layer.
  init(_ underlying: Underlying) {
    self.underlying = underlying
  }

  /// The underlying layer, type-erased to `Any`.
  override var typeErasedBase: Any {
    return underlying
  }

  /// Returns the underlying layer unboxed to the given type, if possible.
  override func unboxed<U: Layer>(to type: U.Type) -> U?
  where U.TangentVector.VectorSpaceScalar == Underlying.TangentVector.VectorSpaceScalar {
    return (self as? ConcreteLayerBox<U>)?.underlying
  }

  // `Differentiable` requirements.
  override func _move(along direction: AnyLayerTangentVector<Underlying.TangentVector.VectorSpaceScalar>) {
    if let scalarDirection = direction.box.getOpaqueScalar() {
      underlying.move(along: Underlying.TangentVector.zero.adding(scalarDirection))
    } else {
      guard let directionBase =
        direction.unboxed(as: Underlying.TangentVector.self) else {
        derivativeTypeMismatch(got: type(of: direction.box.typeErasedBase), expected: Underlying.self)
      }
      underlying.move(along: directionBase)
    }
  }

  // `EuclideanDifferentiable` requirements.
  public override var _differentiableVectorView: AnyLayerTangentVector<Underlying.TangentVector.VectorSpaceScalar> {
    return AnyLayerTangentVector(underlying.differentiableVectorView)
  }

  // `Layer` requirements.
  override func _callAsFunction(_ input: Underlying.Input) -> Underlying.Output {
    return underlying.callAsFunction(input)
  }

  // A helper to group together the model an input since we need a pullback with respect to both.
  struct ModelAndInput: Differentiable {
    var model: Underlying
    var input: Underlying.Input
  }

  override func _vjpCallAsFunction(_ input: Underlying.Input) -> (
    value: Underlying.Output,
    pullback: (Underlying.Output.TangentVector) ->
      (AnyLayerTangentVector<Underlying.TangentVector.VectorSpaceScalar>, Underlying.Input.TangentVector)
  ) {
    let basePullback = valueWithPullback(
      at: ModelAndInput(model: underlying, input: input),
      in: { pair in pair.model.callAsFunction(pair.input) }
    )
    
    return (
      value: basePullback.value,
      pullback: { (outTangent) in
        let pairTangent = basePullback.pullback(outTangent)
        return (
          AnyLayerTangentVector<Underlying.TangentVector.VectorSpaceScalar>(pairTangent.model),
          pairTangent.input
        )
      }
    )
  }

  // `CopyableToDevice` requirements.
  override func _copyToDevice(to device: Device) ->
    AnyLayerBox<Underlying.Input, Underlying.Output, Underlying.TangentVector.VectorSpaceScalar> {
    return ConcreteLayerBox(Underlying(copying: underlying, to: device))
  }

  override func duplicate() ->
    AnyLayerBox<Underlying.Input, Underlying.Output, Underlying.TangentVector.VectorSpaceScalar> {
    return ConcreteLayerBox(underlying)
  }
}

/// A type-erased layer.
///
/// The `AnyLayer` type forwards its operations to an arbitrary underlying
/// value conforming to `Layer`, hiding the specifics of the underlying value.
///
/// This erased layer does not implement `KeyPathIterable` due to a Swift constraint that makes it impossible to
/// cast within a keypath (necessary because the layer is stored as an erased `Any` value). The layer _does_ support
/// `CopyableToDevice`, however, so it can be moved between devices.
///
/// The tangent vector of this type is also type-erased, using the `AnyLayerTangentVector` type. All tangents
/// (other than `zero` and `one`) wrap the tangent vector type of the underlying layer.
///
/// Type Parameters:
///   - Input: the input type of the underlying layar
///   - Output: the output type of the underlying layer
///   - Scalar: the scalar type of the underlying tangent vector
public struct AnyLayer<Input: Differentiable, Output: Differentiable, Scalar: FloatingPoint & ElementaryFunctions>: CopyableToDevice {
  internal var box: AnyLayerBox<Input, Output, Scalar>

  internal init(box: AnyLayerBox<Input, Output, Scalar>) {
    self.box = box
  }

  /// The underlying layer.
  public var underlying: Any {
    return box.typeErasedBase
  }

  /// Creates a type-erased derivative from the given layer.
  @differentiable
  public init<Underlying: Layer>(_ layer: Underlying)
  where Underlying.Input == Input, Underlying.Output == Output, Underlying.TangentVector.VectorSpaceScalar == Scalar {
    self.box = ConcreteLayerBox<Underlying>(layer)
  }

  public init(copying other: AnyLayer, to device: Device) {
    self.box = other.box._copyToDevice(to: device)
  }

  @inlinable
  @derivative(of: init)
  internal static func _vjpInit<T: Layer>(
    _ base: T
  ) -> (value: AnyLayer, pullback: (AnyLayerTangentVector<Scalar>) -> T.TangentVector)
  where T.Input == Input, T.Output == Output, T.TangentVector.VectorSpaceScalar == Scalar
  {
    return (AnyLayer<Input, Output, Scalar>(base), { v in v.unboxed(as: T.TangentVector.self)! })
  }

  @inlinable
  @derivative(of: init)
  internal static func _jvpInit<T: Layer>(
    _ base: T
  ) -> (
    value: AnyLayer, differential: (T.TangentVector) -> AnyLayerTangentVector<Scalar>
  ) where T.Input == Input, T.Output == Output, T.TangentVector.VectorSpaceScalar == Scalar {
    return (AnyLayer<Input, Output, Scalar>(base), { dbase in AnyLayerTangentVector<Scalar>(dbase) })
  }
}

extension AnyLayer: Differentiable {
  public typealias TangentVector = AnyLayerTangentVector<Scalar>

  public mutating func move(along direction: TangentVector) {
    if !isKnownUniquelyReferenced(&box) { // preserve value semantics
      self.box = box.duplicate()
    }
    
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
  func _vjpCallAsFunction(_ input: Input) ->
    (value: Output, pullback: (Output.TangentVector) -> (AnyLayerTangentVector<Scalar>, Input.TangentVector)) {
    return box._vjpCallAsFunction(input)
  }

  @differentiable
  public func callAsFunction(_ input: Input) -> Output {
    return _callAsFunction(input)
  }
}
