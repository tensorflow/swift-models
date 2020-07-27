
/* Modified copy of a subset of base Structural proposal from
https://github.com/google/swift-structural/blob/master/Sources/StructuralCore/Structural.swift

(Enum support excluded.)

Key modifications:
 - Add extra type parameter to StructuralStruct, [StructuralCons], and StructuralParameter
 - Mapping between Structural & StaticStructural representation.
 - Only use `StructuralEmpty` for types with zero entries, and `StructuralCons` when a type has more than one field. (Not seen here.)
*/


/// A type that can be converted to and from its structural representation.
public protocol Structural {
    /// A static representation of `Self`.
    associatedtype StaticStructuralRepresentation: BaseTypeProtocol where StaticStructuralRepresentation.BaseType == Self
    /// A static representation of `Self`.
    static var staticStructuralRepresentation: StaticStructuralRepresentation { get }

    /// A structural representation for `Self`.
    associatedtype StructuralRepresentation: StaticStructuralCorrespondance & BaseTypeProtocol where StructuralRepresentation.StaticStructuralType == StaticStructuralRepresentation, StructuralRepresentation.BaseType == Self

    /// Creates an instance from the given structural representation.
    init(structuralRepresentation: StructuralRepresentation)

    /// A structural representation of `self`.
    var structuralRepresentation: StructuralRepresentation { get set }
}

/// Structural representation of a Swift struct.
public struct StructuralStruct<BaseType, Properties> {
    public var type: Any.Type?
    public var properties: Properties

    public init(_ properties: Properties) {
        self.type = nil
        self.properties = properties
    }

    public init(_ type: Any.Type, _ properties: Properties) {
        self.type = type
        self.properties = properties
    }
}

/// Structural representation of a heterogeneous list cons cell.
public struct StructuralCons<Value, Next> {
    public var value: Value
    public var next: Next

    public init(_ value: Value, _ next: Next) {
        self.value = value
        self.next = next
    }
}

/// Structural representation of a Swift property.
public struct StructuralProperty<BaseType, Value> {
    public var name: String
    public var value: Value
    public var isMutable: Bool

    public init(_ value: Value) {
        self.name = ""
        self.value = value
        self.isMutable = false
    }

    public init(_ name: String, _ value: Value) {
        self.name = name
        self.value = value
        self.isMutable = false
    }

    public init(_ name: String, _ value: Value, isMutable: Bool) {
        self.name = name
        self.value = value
        self.isMutable = isMutable
    }
}

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/// StaticStructuralCorrespondance protocol & BaseType protocol.
/////////////////////////////////////////////////////////////////////////////////////////


/// Defines mapping from instance to static structural representations.
public protocol StaticStructuralCorrespondance {
    associatedtype StaticStructuralType
}

extension StructuralStruct: StaticStructuralCorrespondance where Properties: StaticStructuralCorrespondance {
    public typealias StaticStructuralType = StaticStructuralStruct<BaseType, Properties.StaticStructuralType>
}

extension StructuralCons: StaticStructuralCorrespondance where Value: StaticStructuralCorrespondance, Next: StaticStructuralCorrespondance {
    public typealias StaticStructuralType = StructuralCons<Value.StaticStructuralType, Next.StaticStructuralType>
}

extension StructuralProperty: StaticStructuralCorrespondance {
    public typealias StaticStructuralType = StaticStructuralProperty<BaseType, Value>
}

public protocol BaseTypeProtocol {
    associatedtype BaseType
}

extension StructuralStruct: BaseTypeProtocol where Properties: BaseTypeProtocol, Properties.BaseType == BaseType {}
extension StructuralCons: BaseTypeProtocol where Value: BaseTypeProtocol, Next: BaseTypeProtocol, Value.BaseType == Next.BaseType {
    public typealias BaseType = Value.BaseType
}
extension StructuralProperty: BaseTypeProtocol {}

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/// StaticStructural extension
/////////////////////////////////////////////////////////////////////////////////////////

// TODO: Need correspondance the other way in order to make the full sequential construction work!
// Note: this also requires changing the Structural types to take an additional type parameter!

public struct StaticStructuralStruct<BaseType, Properties>: BaseTypeProtocol {

    public var type: BaseType.Type

    public var properties: Properties
}

public struct StaticStructuralProperty<BaseType, Value>: BaseTypeProtocol {

    public let keyPath: KeyPath<BaseType, Value>
    public var name: String
    public var isMutable: Bool
}
