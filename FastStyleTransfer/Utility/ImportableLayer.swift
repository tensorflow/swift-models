import TensorFlow
import Python
let np = Python.import("numpy")

/// Adds ability to recursively override Layer properties
/// WARNING This is temporary solution it relies on the fact that Mirror and KeyPathIterable enumerate properties in the same order

public protocol ImportableLayer: KeyPathIterable {}

public typealias ImportMap = [String: (String, [Int]?)]

public extension ImportableLayer {

    /// List all properties of specified type
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

    /// Dict of property name => property key path
    func getRecursiveNamedKeyPaths<T>(ofType valueType: T.Type) -> [String: WritableKeyPath<Self, T>] {
        let labels = getRecursiveProperties(ofType: T.self)
        let keys = self.recursivelyAllWritableKeyPaths(to: T.self)
        return Dictionary(uniqueKeysWithValues: zip(labels, keys))
    }

    /// Updates model with parameters recursively, according to map:
    /// [ model_property_name : ( param_name, tensor_permutations? ) ]
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

    mutating func unsafeImport(fromNumpyArchive file: String, map: [String: (String, [Int]?)]) {
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
