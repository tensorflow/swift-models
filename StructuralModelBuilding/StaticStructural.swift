import StructuralCore

public protocol StaticStructural: Structural {
    associatedtype StaticStructuralRepresentation: StaticStructuralCorrespondance where StaticStructuralRepresentation.StructuralType == StructuralRepresentation
    // TODO: Refine StructuralRepresentation type to have mapping the other direction too!

    static var staticStructuralRepresentation: StaticStructuralRepresentation { get }
}

/// Defines mapping from static to instance structural representations.
public protocol StaticStructuralCorrespondance {
    associatedtype StructuralType
}

// TODO: Need correspondance the other way in order to make the full sequential construction work!
// Note: this also requires changing the Structural types to take an additional type parameter!

public struct StaticStructuralStruct<BaseType, Properties: StaticStructuralCorrespondance>: StaticStructuralCorrespondance {
    public typealias StructuralType = StructuralStruct<Properties.StructuralType>

    public var type: BaseType.Type

    public var properties: Properties
}

public struct StaticStructuralProperty<BaseType, Value>: StaticStructuralCorrespondance {
    public typealias StructuralType = StructuralProperty<Value>

    public let keyPath: KeyPath<BaseType, Value>
    public var name: String
    public var isMutable: Bool
}

extension StructuralCons: StaticStructuralCorrespondance where Value: StaticStructuralCorrespondance, Next: StaticStructuralCorrespondance {
    public typealias StructuralType = StructuralCons<Value.StructuralType, Next.StructuralType>
}

extension StructuralEmpty: StaticStructuralCorrespondance {
    public typealias StructuralType = StructuralEmpty
}

// TODO: Some sort of correspondance between static & dynamic types?
