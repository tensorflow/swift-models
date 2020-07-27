import TensorFlow
import PenguinStructures

// TODO: Pick a better name.
// TODO: Consider splitting the inductive cases into a separate protocol.
/// A layer composed of a sequential application of its constituent field layers.
public protocol SequentialLayer: Differentiable {
    associatedtype SequentialInput: Differentiable  // TODO: support embedding layers.
    associatedtype SequentialOutput: Differentiable

    @differentiable(wrt: (self, input))
    func sequentialApply(_ input: SequentialInput) -> SequentialOutput
}

extension SequentialLayer
where
    Self: DifferentiableStructural & Layer,
    Self.StructuralRepresentation: SequentialLayer,
    SequentialInput == StructuralRepresentation.SequentialInput,
    SequentialOutput == StructuralRepresentation.SequentialOutput,
    SequentialInput == Input,
    SequentialOutput == Output
{
    @differentiable
    public func sequentialApply(_ input: SequentialInput) -> SequentialOutput {
        self.differentiableStructuralRepresentation.sequentialApply(input)
    }
    @differentiable
    public func callAsFunction(_ input: Input) -> Output {
        self.sequentialApply(input)
    }
}

extension StructuralCons: SequentialLayer where Value: SequentialLayer, Next: SequentialLayer, Next.SequentialInput == Value.SequentialOutput {

    public typealias SequentialInput = Value.SequentialInput
    public typealias SequentialOutput = Next.SequentialOutput

    @differentiable
    public func sequentialApply(_ input: SequentialInput) -> SequentialOutput {
        let tmp = value.sequentialApply(input)
        return next.sequentialApply(tmp)
    }
}

extension StructuralProperty: SequentialLayer where Value: Layer {
    public typealias SequentialInput = Value.Input
    public typealias SequentialOutput = Value.Output

    @differentiable
    public func sequentialApply(_ input: SequentialInput) -> SequentialOutput { value(input) }
}

extension StructuralStruct: SequentialLayer where Properties: SequentialLayer {
    public typealias SequentialInput = Properties.SequentialInput
    public typealias SequentialOutput = Properties.SequentialOutput
    @differentiable
    public func sequentialApply(_ input: SequentialInput) -> SequentialOutput { properties.sequentialApply(input) }
}

/// Allows skipping a field in a SequentialLayer
///
/// - SeeAlso: `SequentialLayer`.
@propertyWrapper
public struct SequentialSkip<Underlying, PassingType: Differentiable>: KeyPathIterable {
    public var wrappedValue: Underlying

    public init(wrappedValue: Underlying, passing passingType: Type<PassingType>) {
        self.wrappedValue = wrappedValue
    }
}

extension SequentialSkip: Differentiable {
    // public typealias TangentVector = Empty
    public typealias TangentVector = Empty2
    public mutating func move(along direction: TangentVector) {}
    public var zeroTangentVectorInitializer: () -> TangentVector { { () in TangentVector() } }
}

extension SequentialSkip: EuclideanDifferentiable {
    public var differentiableVectorView: Self.TangentVector { TangentVector() }
}

// See error in `StructuralDifferentiability.swift` regarding `ElementaryFunctions` conformance for `Empty`.
extension SequentialSkip: Layer {
    public typealias SequentialInput = PassingType
    public typealias SequentialOutput = PassingType

    @differentiable
    public func callAsFunction(_ input: SequentialInput) -> SequentialOutput { input }
}

// Work around compiler error regarding retroactive conformance to `ElementaryFunctions` (See StructuralDifferentability.)
public struct Empty2: Differentiable, EuclideanDifferentiable, KeyPathIterable, PointwiseMultiplicative, ElementaryFunctions, VectorProtocol {
    public typealias VectorSpaceScalar = Float
    public func adding(_ x: Self.VectorSpaceScalar) -> Self { Self() }
    public func subtracting(_ x: Self.VectorSpaceScalar) -> Self { Self() }
    public func scaled(by scalar: Self.VectorSpaceScalar) -> Self { Self() }
}

// TODO: Make merge function configurable?
@propertyWrapper
public struct ResidualConnection<Underlying: Layer>: Layer where Underlying.Input == Underlying.Output, Underlying.Input == Tensor<Float> {  // TODO: generalize beyond Tensor<Float>!
    public var wrappedValue: Underlying

    public init(wrappedValue: Underlying) {
        self.wrappedValue = wrappedValue
    }

    @differentiable
    public func callAsFunction(_ input: Underlying.Input) -> Underlying.Output {
        let tmp = wrappedValue(input)
        return input + tmp
    }
}
