import TensorFlow
import Python
let np = Python.import("numpy")

public protocol ImportableLayer: KeyPathIterable {}

/// Map between model parameters and weights dictionary contents.
/// 
/// Where:
///  - Key: property name in the model. Dot indicates properties nesting.
///  - Value: tuple of key name in the data dictionary and data permutation to apply to data tensor.
public typealias ImportMap = [String: (String, [Int]?)]

/// Adds ability to set Layer properties recursively.
///
/// WARNING: This is temporary solution for importing weights, 
/// it relies on the fact that Mirror and KeyPathIterable 
/// enumerate properties in the same order.
public extension ImportableLayer {
    /// Returns list of names of all recursive properties of the specified `valueType`.
    /// Properties nesting is identified by `.`.
    func getRecursiveProperties<T>(ofType valueType: T.Type) -> [String] {
        func _get(_ obj: Any, _ parent: String? = nil, out: inout [String]) {
            let m = Mirror(reflecting: obj)
            for child in m.children {
                let keypath = (parent != nil ? parent! + "." : "") + child.label!
                if (child.value is T) {
                    out.append(keypath)
                }
                _get(child.value, keypath, out: &out)
            }
        }

        var labels = [String]()
        _get(self, out: &labels)
        return labels
    }

    /// Returns map of all recursive properties name to keypath.
    func getRecursiveNamedKeyPaths<T>(
            ofType valueType: T.Type
    ) -> [String: WritableKeyPath<Self, T>] {
        let labels = getRecursiveProperties(ofType: T.self)
        let keys = self.recursivelyAllWritableKeyPaths(to: T.self)
        return Dictionary(uniqueKeysWithValues: zip(labels, keys))
    }

    /// Updates model parameters with values from `parameters`, according to `ImportMap`.
    /// TODO add shapes check
    mutating func unsafeImport<T>(parameters: [String: Tensor<T>], map: ImportMap) {
        for (label, keyPath) in getRecursiveNamedKeyPaths(ofType: Tensor<T>.self) {
            // let shape = self[keyPath: keyPath].shape
            if let mapping = map[label], var weights = parameters[mapping.0] {
                if let permutes = mapping.1 {
                    weights = weights.transposed(withPermutations: permutes)
                }
                self[keyPath: keyPath] = weights
                // print("imported \(mapping.0) \(shape) -> \(label) \(weights.shape)")
            } else if let weights = parameters[label] {
                self[keyPath: keyPath] = weights
                // print("imported \(label) \(shape) -> \(label) \(weights.shape)")
            }
        }
    }
}

public extension ImportableLayer {
    /// Updates model parameters with values from numpy archive, according to `ImportMap`.
    mutating func unsafeImport(fromNumpyArchive file: String, map: ImportMap) {
        let data = np.load(file)
        var parameters = [String: Tensor<Float>]()
        for label in data.files {
            if let label = String(label) {
                parameters[label] = Tensor<Float>(numpy: data[label])
            }
        }
        unsafeImport(parameters: parameters, map: map)
    }
}
