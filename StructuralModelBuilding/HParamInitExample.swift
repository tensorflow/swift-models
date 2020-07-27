import TensorFlow
import StructuralCore
import PenguinStructures



public struct MyInitModel {
    var conv: Conv2D<Float>
    var flatten: Flatten<Float>
    var dense: Dense<Float>
}
// See below for explicit conformances to `DifferentiableStructural` and `StaticKeyPathIterable`

// Thanks to `DifferentiableStructural` conformances, we can derive these protocols automagically!
// extension MyInitModel: HParamInitLayer, Layer, SequentialLayer {
//     public typealias HParam = StructuralRepresentation.HParam
// }

func sampleModelUsage() -> MyInitModel {
    fatalError("TODO: WRITE ME OUT!")
}

public struct MyInitModelExplicit: HParamInitLayer, StaticKeyPathIterable {
    var conv: Conv2D<Float>
    var flatten: Flatten<Float>
    var dense: Dense<Float>

    public typealias StaticKeyPaths = [PartialKeyPath<Self>]
    public static var staticKeyPaths = [
        \Self.conv,
        \Self.flatten,
        \Self.dense,
    ]

    public typealias HParam = StructuralHParams<Self,
        HParamCons<Self, HParamHolder<Self, Conv2D<Float>>,
        HParamCons<Self, HParamHolder<Self, Flatten<Float>>,
            HParamHolder<Self, Dense<Float>>>>>
    
    public init(hparam: HParam, inputExample: Tensor<Float>) {
        var tmp = inputExample
        conv = .init(hparam: hparam.conv!, inputExample: tmp)
        tmp = conv(tmp)  // Move forward.
        flatten = .init(hparam: hparam.flatten!, inputExample: tmp)
        tmp = flatten(tmp)  // Move forward.
        dense = .init(hparam: hparam.dense!, inputExample: tmp)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        dense(flatten(conv(input)))
    }
}

func makeExplicitModel() -> MyInitModelExplicit {
    var hparams = MyInitModelExplicit.HParam()
    hparams.conv = .init(height: 3, width: 3, channels: 10)  // Fully typesafe!
    hparams.dense = .init(size: 10)

    return hparams.build(for: Tensor<Float>(zeros: [5, 28, 28, 1]))
}

// TODO: Figure out how StaticKeyPathIterable's capabilities can be constructed just from `Structural`.
extension MyInitModel: StaticKeyPathIterable {
    public typealias StaticKeyPaths = [PartialKeyPath<Self>]
    public static var staticKeyPaths = [
        \Self.conv,
        \Self.flatten,
        \Self.dense,
    ]
}

extension MyInitModel: DifferentiableStructural {

    public typealias StructuralRepresentation =
        StructuralStruct<
            StructuralCons<StructuralProperty<Conv2D<Float>>,
            StructuralCons<StructuralProperty<Flatten<Float>>,
            StructuralCons<StructuralProperty<Dense<Float>>,
            StructuralEmpty>>>>

    @differentiable
    public init(differentiableStructuralRepresentation: StructuralRepresentation) {
        fatalError()
    }

    @derivative(of: init(differentiableStructuralRepresentation:))
    public static func _vjp_init(differentiableStructuralRepresentation: StructuralRepresentation)
    -> (value: Self, pullback: (TangentVector) -> StructuralRepresentation.TangentVector)
    {
        fatalError()
    }

    @differentiable
    public var differentiableStructuralRepresentation: StructuralRepresentation {
        fatalError()
    }

    @derivative(of: differentiableStructuralRepresentation)
    public func _vjp_differentiableStructuralRepresentation()
    -> (value: StructuralRepresentation, pullback: (StructuralRepresentation.TangentVector) -> TangentVector)
    {
        fatalError()
    }
}
