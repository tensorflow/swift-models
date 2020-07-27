import PenguinStructures
import TensorFlow

// TODO: Pick a better name.
/// A layer that propagates shapes during initialization.
public protocol ShapePropagatingLayer: SequentialLayer {
    // associatedtype ShapeTracker: FixedSizeArray  // TODO: generalize shape tracking?

    /// Initializes `self`, updating `shapeTracker` to reflect the new output shape.
    init(shapeTracker: inout Tensor<Float>)  // TODO: different shape tracking type?
}

extension ShapePropagatingLayer
where
    Self: DifferentiableStructural & Layer,
    Self.StructuralRepresentation: ShapePropagatingLayer,
    SequentialInput == StructuralRepresentation.SequentialInput,
    SequentialOutput == StructuralRepresentation.SequentialOutput,
    SequentialInput == Input,
    SequentialOutput == Output
{
    public init(shapeTracker: inout Tensor<Float>) {
        self.init(structuralRepresentation: StructuralRepresentation(shapeTracker: &shapeTracker))
    }
}

// Inductive cases

extension StructuralCons: ShapePropagatingLayer
where
    Value: ShapePropagatingLayer,
    Next: ShapePropagatingLayer,
    Next.SequentialInput == Value.SequentialOutput
{
    public init(shapeTracker: inout Tensor<Float>) {
        let value = Value(shapeTracker: &shapeTracker)
        let next = Next(shapeTracker: &shapeTracker)
        self.init(value, next)
    }
}

extension StructuralProperty: ShapePropagatingLayer where Value: ShapePropagatingLayer & Layer {
    public init(shapeTracker: inout Tensor<Float>) {
        self.init(Value(shapeTracker: &shapeTracker))
    }
}

extension StructuralStruct: ShapePropagatingLayer where Properties: ShapePropagatingLayer {
    public init(shapeTracker: inout Tensor<Float>) {
        self.init(Properties(shapeTracker: &shapeTracker))
    }
}

// TODO: HParam property wrapper?
