import TensorFlow
import StructuralCore
import PenguinStructures

/// A layer that can be initialized with some hyperparameters and a representative input.
///
/// The HParam represents hyperparameters to the layer, and the input example encodes shape
/// information / etc.
public protocol HParamInitLayer: Layer {
    /// Hyper parameters for `self`.
    associatedtype HParam

    init(hparam: HParam, inputExample: Input)
}

/// Like KeyPathIterable, but available statically!
public protocol StaticKeyPathIterable {
    associatedtype StaticKeyPaths: Collection where StaticKeyPaths.Element == PartialKeyPath<Self>
    static var staticKeyPaths: StaticKeyPaths { get }
}

// Retroactive conformances.

extension Conv2D: HParamInitLayer {
    public struct HParam {
        public init(height: Int, width: Int, channels: Int) {
            self.height = height
            self.width = width
            self.channels = channels
        }

        /// The height of the filter.
        var height: Int
        /// The width of the filter.
        var width: Int
        /// The number of output channels.
        var channels: Int

        // The number of input channels is automatically inferred from the provided input tensor.

        /// The strides of the sliding window for spatial dimensions.
        var strides = (1, 1)

        var padding: Padding = .valid

        // TODO: Add others (e.g. initializers, usebias, dialation, etc).
    }

    public init(hparam: HParam, inputExample: Tensor<Scalar>) {
        precondition(inputExample.shape.count == 4)
        let inputChannelCount = inputExample.shape[3]  // Assumes channels last.

        self.init(
            filterShape: (hparam.height, hparam.width, inputChannelCount, hparam.channels),
            strides: hparam.strides,
            padding: hparam.padding)
    }
}

extension Flatten: HParamInitLayer {
    public typealias HParam = Empty

    public init(hparam: HParam, inputExample: Tensor<Scalar>) {
        self.init()
    }
}

extension Dense: HParamInitLayer {
    public struct HParam {
        public init(size: Int) {
            self.size = size
        }

        /// The output size of the Dense layer.
        var size: Int

        // Input size comes from the inputExample.
        // TODO: Add extra hparams here!
    }

    public init(hparam: HParam, inputExample: Tensor<Scalar>) {
        precondition(inputExample.shape.count == 2, "input example must be a matrix, got: \(inputExample.shape)")
        let inputSize = inputExample.shape[1]  // [batch, inputSize]

        self.init(inputSize: inputSize, outputSize: hparam.size)
    }
}

public protocol _HParamHolderProtocol {
    associatedtype Model: StaticKeyPathIterable

    init(keyPaths: Model.StaticKeyPaths, index: Model.StaticKeyPaths.Index)

    func getHParam<V: HParamInitLayer>(keyPath: KeyPath<Model, V>) -> V.HParam?
    mutating func storeHParam<V: HParamInitLayer>(keyPath: KeyPath<Model, V>, value: V.HParam)
}

/// A struct that holds an optionally initialized hyperparameter for a constitutent layer.
public struct HParamHolder<Model: StaticKeyPathIterable, Layer: HParamInitLayer>: _HParamHolderProtocol {
    /// The keyPath we use 
    var keyPath: KeyPath<Model, Layer>
    var value: Layer.HParam?

    public init(_ keyPath: KeyPath<Model, Layer>) {
        self.keyPath = keyPath
    }

    public init(keyPaths: Model.StaticKeyPaths, index: Model.StaticKeyPaths.Index) {
        let pkp = keyPaths[index]
        guard let kp = pkp as? KeyPath<Model, Layer> else {
            preconditionFailure("Key path \(pkp) at index \(index) in keyPaths \(keyPaths) not of expected type: \(KeyPath<Model, Layer>.self).")
        }
        self.keyPath = kp
    }

    public func getHParam<V: HParamInitLayer>(keyPath: KeyPath<Model, V>) -> V.HParam? {
        if keyPath == self.keyPath {
            if let value = value {
                return (value as! V.HParam)
            } else {
                // TODO: if V.HParam: DefaultInitializable, return that instead of crashing!
                preconditionFailure("HParam for \(keyPath) has not yet been initialized!")
            }
        } else {
            return nil
        }
    }

    public mutating func storeHParam<V: HParamInitLayer>(keyPath: KeyPath<Model, V>, value: V.HParam) {
        if keyPath == self.keyPath {
            self.value = (value as! Layer.HParam)  // Force downcast to catch errors.
        }
    }
}

/// The Cons-list of `HParamHolder`s.
public struct HParamCons<Model, Value: _HParamHolderProtocol, Next: _HParamHolderProtocol>: _HParamHolderProtocol where Value.Model == Model, Next.Model == Model {
    var value: Value
    var next: Next

    public init(keyPaths: Model.StaticKeyPaths, index: Model.StaticKeyPaths.Index) {
        value = .init(keyPaths: keyPaths, index: index)
        let nextIndex = keyPaths.index(after: index)
        next = .init(keyPaths: keyPaths, index: nextIndex)
    }

    public mutating func storeHParam<V: HParamInitLayer>(keyPath: KeyPath<Model, V>, value: V.HParam) {
        self.value.storeHParam(keyPath: keyPath, value: value)
        next.storeHParam(keyPath: keyPath, value: value)
    }

    public func getHParam<V: HParamInitLayer>(keyPath: KeyPath<Model, V>) -> V.HParam? {
        value.getHParam(keyPath: keyPath) ?? next.getHParam(keyPath: keyPath)
    }
}

@dynamicMemberLookup
public struct StructuralHParams<Layer: HParamInitLayer /*& StaticKeyPathIterable*/, Values: _HParamHolderProtocol>: DefaultInitializable where Values.Model == Layer {
    public init() {
        let keyPaths = Layer.staticKeyPaths
        values = .init(keyPaths: keyPaths, index: keyPaths.startIndex)
    }

    var values: Values

    subscript<T: HParamInitLayer>(dynamicMember keyPath: KeyPath<Layer, T>) -> T.HParam? {
        get { values.getHParam(keyPath: keyPath) }
        set {
            guard let newValue = newValue else { fatalError("Cannot unset hparam values! (Attempted to unset keyPath: \(keyPath)") }
            values.storeHParam(keyPath: keyPath, value: newValue)
        }
    }
}

extension StructuralHParams where Layer.HParam == Self {
    public func build(for exampleInput: Layer.Input) -> Layer {
        return .init(hparam: self, inputExample: exampleInput)
    }
}


// Build type using StaticStructural!

extension StructuralCons: _HParamHolderProtocol where Value: _HParamHolderProtocol, Next: _HParamHolderProtocol, Value.Model == Next.Model {
    public typealias Model = Value.Model

    public init(keyPaths: Model.StaticKeyPaths, index: Model.StaticKeyPaths.Index) {
        let nextIndex = keyPaths.index(after: index)
        self.init(
            .init(keyPaths: keyPaths, index: index),
            .init(keyPaths: keyPaths, index: nextIndex))
    }

    public mutating func storeHParam<V: HParamInitLayer>(keyPath: KeyPath<Model, V>, value: V.HParam) {
        self.value.storeHParam(keyPath: keyPath, value: value)
        next.storeHParam(keyPath: keyPath, value: value)
    }

    public func getHParam<V: HParamInitLayer>(keyPath: KeyPath<Model, V>) -> V.HParam? {
        value.getHParam(keyPath: keyPath) ?? next.getHParam(keyPath: keyPath)
    }
}

public protocol _HParamHolderStructural {
    associatedtype HParam: _HParamHolderProtocol

    func makeEmptyHParam() -> HParam
}

extension StaticStructuralStruct where Properties: _HParamHolderStructural, BaseType: HParamInitLayer, Properties.HParam.Model == BaseType {
    public typealias HParam = StructuralHParams<BaseType, Properties.HParam>

    public func makeEmptyHParam() -> HParam {
        .init()
    }
}

extension StaticStructuralProperty: _HParamHolderStructural where BaseType: StaticKeyPathIterable, Value: HParamInitLayer {
    public typealias HParam = HParamHolder<BaseType, Value>

    public func makeEmptyHParam() -> HParam {
        .init(keyPath)
    }
}

