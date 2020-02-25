// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import TensorFlow

// Private protocol used to derive conformance to Collatable using KeyPathIterable
public protocol _Collatable {
     static func _collateLeaf<Root>(
         _ rootOut: inout Root, _ rootKeyPath: PartialKeyPath<Root>, _ rootIn: [Root])
}

// Collatable: a protocol representing a type where you can stack elements together to
// get some higher-rank element of the same type (example: tensors, tuple of tensors)
public protocol Collatable: _Collatable {
    init(collating: [Self])
}

// For derived conformance
extension Collatable {
    public static func _collateLeaf<Root>(
        _ rootOut: inout Root, _ rootKeyPath: PartialKeyPath<Root>, _ rootIn: [Root]
    ) {
        guard let keyPath = rootKeyPath as? WritableKeyPath<Root, Self> else {
            fatalError("Failed conversion from \(rootKeyPath) to 'WritableKeyPath<\(Root.self), \(Self.self)>'")
        }
        rootOut[keyPath: keyPath] = Self.init(collating: rootIn.map { $0[keyPath: keyPath] })
    }
}

// For derived conformance
extension _KeyPathIterableBase {
    public func _collateAll<Root>(
        _ rootOut: inout Root, _ rootKeyPath: PartialKeyPath<Root>, _ rootIn: [Root]) {
        for kp in _allKeyPathsTypeErased {
            let joinedKeyPath = rootKeyPath.appending(path: kp)!
            if let valueType = type(of: joinedKeyPath).valueType as? _Collatable.Type {
                valueType._collateLeaf(&rootOut, joinedKeyPath, rootIn)
            } else if let nested = self[keyPath: kp] as? _KeyPathIterableBase {
                nested._collateAll(&rootOut, joinedKeyPath, rootIn)
            } else {
                fatalError("Key path \(kp) is not Collatable")
            }
        }
    }
}

// For derived conformance
extension KeyPathIterable {
    public init(collating roots: [Self]) {
        self = roots[0]
        _collateAll(&self, \.self, roots)
    }
}

// Tensor are collated using stacking
extension Tensor: Collatable {
    public init(collating: [Self]) { self.init(stacking: collating) }
}

// Example: you can derive conformance to Collatable directly if a struct has only tensors
// struct Pair : Collatable, KeyPathIterable {
//     var first: Tensor
//     var second: Tensor
//     var third: Tensor = Tensor(5.0)
// }
