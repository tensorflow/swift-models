import StructuralCore
import PenguinStructures

public protocol DifferentiableStructural: Structural where Self: Differentiable, StructuralRepresentation: Differentiable {

    @differentiable
    init(differentiableStructuralRepresentation: StructuralRepresentation)

    @differentiable
    var differentiableStructuralRepresentation: StructuralRepresentation { get }
}

extension DifferentiableStructural {
    public var structuralRepresentation: StructuralRepresentation { differentiableStructuralRepresentation }

    public init(structuralRepresentation: StructuralRepresentation) {
        self.init(differentiableStructuralRepresentation: structuralRepresentation)
    }
}

extension StructuralCons: Differentiable where Value: Differentiable, Next: Differentiable {

    public struct TangentVector: Differentiable & AdditiveArithmetic {
        var value: Value.TangentVector
        var next: Next.TangentVector
    }

    public mutating func move(along direction: TangentVector) {
        value.move(along: direction.value)
        next.move(along: direction.next)
    }

    public var zeroTangentVectorInitializer: () -> TangentVector {
        let valueZeroInit = value.zeroTangentVectorInitializer
        let nextZeroInit = next.zeroTangentVectorInitializer
        return { () in TangentVector(value: valueZeroInit(), next: nextZeroInit()) }
    }
}

extension StructuralProperty: Differentiable where Value: Differentiable {
    public struct TangentVector: Differentiable & AdditiveArithmetic {
        var value: Value.TangentVector
    }

    public mutating func move(along direction: TangentVector) {
        value.move(along: direction.value)
    }

    public var zeroTangentVectorInitializer: () -> TangentVector {
        let valueInit = value.zeroTangentVectorInitializer
        return { () in TangentVector(value: valueInit()) }
    }
}

extension StructuralEmpty: AdditiveArithmetic {
    public static func == (lhs: Self, rhs: Self) -> Bool { true }
    public static var zero: Self { Self() }
    public static func + (lhs: Self, rhs: Self) -> Self { Self() }
    public static func += (lhs: inout Self, rhs: Self) {}
    public static func - (lhs: Self, rhs: Self) -> Self { Self() }
    public static func -= (lhs: inout Self, rhs: Self) {}
}

extension StructuralEmpty: Differentiable {
    public typealias TangentVector = Self
    public mutating func move(along direction: TangentVector) {}
    public var zeroTangentVectorInitializer: () -> TangentVector { { () in Self() } }
}

extension StructuralStruct: Differentiable where Properties: Differentiable {
    public struct TangentVector: Differentiable & AdditiveArithmetic {
        var properties: Properties.TangentVector
    }

    public mutating func move(along direction: TangentVector) {
        properties.move(along: direction.properties)
    }

    public var zeroTangentVectorInitializer: () -> TangentVector {
        let propertiesInit = properties.zeroTangentVectorInitializer
        return { () in TangentVector(properties: propertiesInit()) }
    }
}

// TODO: Add in EuclideanDifferentiable refinements.

/*

/usr/local/google/home/saeta/src/swift-models/StructuralModelBuilding/StructuralDifferentiability.swift:77:1: error: implementation of 'ElementaryFunctions' cannot be automatically synthesized in an extension in a different file to the type
extension Empty: ElementaryFunctions {
^
/usr/local/google/home/saeta/src/swift-models/.build/checkouts/penguin/Sources/PenguinStructures/Empty.swift:16:15: note: type declared here
public struct Empty: DefaultInitializable {
              ^


extension Empty: ElementaryFunctions {
    public static func exp(_ x: Self) -> Self { Self() }
    public static func expMinusOne(_ x: Self) -> Self { Self() }
    public static func cosh(_ x: Self) -> Self { Self() }
    public static func sinh(_ x: Self) -> Self { Self() }
    public static func tanh(_ x: Self) -> Self { Self() }
    public static func cos(_ x: Self) -> Self { Self() }
    public static func sin(_ x: Self) -> Self { Self() }
    public static func tan(_ x: Self) -> Self { Self() }
    public static func log(_ x: Self) -> Self { Self() }
    public static func log(onePlus x: Self) -> Self { Self() }
    public static func acosh(_ x: Self) -> Self { Self() }
    public static func asinh(_ x: Self) -> Self { Self() }
    public static func atanh(_ x: Self) -> Self { Self() }
    public static func acos(_ x: Self) -> Self { Self() }
    public static func asin(_ x: Self) -> Self { Self() }
    public static func atan(_ x: Self) -> Self { Self() }
    public static func pow(_ x: Self, _ y: Self) -> Self { Self() }
    public static func pow(_ x: Self, _ n: Int) -> Self { Self() }
    public static func sqrt(_ x: Self) -> Self { Self() }
    public static func root(_ x: Self, _ n: Int) -> Self { Self() }
}

extension Empty: KeyPathIterable {
    public var allKeyPaths: [PartialKeyPath<Self>] { [] }
}

extension Empty: PointwiseMultiplicative {
    public static var one: Self { Self() }
    public static func * (lhs: Self, rhs: Self) -> Self { Self() }
}

extension Empty: VectorProtocol {
    public typealias VectorSpaceScalar = Float

    public func adding(_ scalar: VectorSpaceScalar) -> Self { Self() }
    public func subtracting(_ scalar: VectorSpaceScalar) -> Self { Self() }
    public func scaled(by scalar: VectorSpaceScalar) -> Self { Self() }
    public func add(_ scalar: VectorSpaceScalar) {}
    public func subtract(_ scalar: VectorSpaceScalar) {}
    public func scale(by scalar: VectorSpaceScalar) {}
}
*/
extension Empty: AdditiveArithmetic {
    public static func == (lhs: Self, rhs: Self) -> Bool { true }
    public static var zero: Self { Self() }
    public static func + (lhs: Self, rhs: Self) -> Self { Self() }
    public static func += (lhs: inout Self, rhs: Self) {}
    public static func - (lhs: Self, rhs: Self) -> Self { Self() }
    public static func -= (lhs: inout Self, rhs: Self) {}
}
extension Empty: Differentiable {
    public typealias TangentVector = Self
    public mutating func move(along direction: TangentVector) {}
    public var zeroTangentVectorInitializer: () -> TangentVector { { () in Self() } }
}
