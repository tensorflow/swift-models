// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

import Foundation
import ModelSupport
import TensorFlow

// TODO: Extend handling to different tensor types.

/// A Swift-native TensorFlow v2 checkpoint writer. This writer has no dependencies
/// on the TensorFlow runtime or libraries.
open class CheckpointWriter {
    let fileSystem: FileSystem

    /// Initializes the checkpoint reader from a dictionary of tensors, keyed on their string names.
    ///
    /// - Parameters:
    ///   - fileSystem: The filesystem used for writing the checkpoint.
    public init(fileSystem: FileSystem = FoundationFileSystem()
    ) {
        self.fileSystem = fileSystem
    }

    /// Writes the checkpoint to disk, in a specified directory. A TensorFlow v2 checkpoint consists
    /// of a directory that contains a [name].index header file and one or more
    /// [name].data-0000X-of-0000Y binary shard files with the tensor bytes within them.
    ///
    /// - Parameters:
    ///   - tensors: The tensors to be written, keyed by the names of those tensors to write in the
    ///     checkpoint.
    ///   - directory: The directory to write the checkpoint into. If it doesn't exist, it will be
    ///     created.
    ///   - name: The base name of the checkpoint, which is what will have the .index and
    ///     .data-0000X-of-0000Y extensions appended to it for files in the checkpoint directory.
    public func write(tensors: [String: Tensor<Float>], to directory: URL, name: String) throws {
        try fileSystem.createDirectoryIfMissing(at: directory.path)
        let indexWriter = CheckpointIndexWriter(tensors: tensors)
        let indexHeader = indexWriter.serializedHeader()
        let headerLocation = directory.appendingPathComponent("\(name).index")
        let headerFile = fileSystem.open(headerLocation.path)
        try headerFile.write(indexHeader)

        // TODO: Handle splitting into multiple shards.
        try writeShard(
            to: directory.appendingPathComponent("\(name)"), shard: 0, numShards: 1,
            tensors: tensors, tensorList: indexWriter.orderedTensors)
    }

    func writeShard(
        to location: URL, shard: Int, numShards: Int, tensors: [String: Tensor<Float>],
        tensorList: [String]
    ) throws {
        let shardFile = CheckpointReader.shardFile(
            location: location, shard: shard, totalShards: numShards)

        var outputBuffer = Data()
        // TODO: Write this directly to disk, rather than accumulating it in memory.
        for tensorName in tensorList {
            print("Writing tensor: \(tensorName)")
            guard let tensor = tensors[tensorName] else {
                fatalError("Mismatch in sorted tensors at name: \(tensorName).")
            }
            let scalars = tensor.array.scalars
            scalars.withUnsafeBufferPointer { (ptr) in
                ptr.baseAddress!.withMemoryRebound(
                    to: UInt8.self, capacity: ptr.count * MemoryLayout<Float>.size
                ) {
                    outputBuffer.append($0, count: ptr.count * MemoryLayout<Float>.size)
                }
            }
        }

        let outputFile = fileSystem.open(shardFile.path)
        try outputFile.write(outputBuffer)
    }
}

extension CheckpointWriter {
    static func recursivelyObtainTensors(
        _ current: Any, scope: String? = nil, tensors: inout [String: Tensor<Float>],
        separator: String, ignoredTensorPaths: Set<String> = []
    ) {
        let currentType = String(describing: type(of: current.self))
        let m = Mirror(reflecting: current)
        
        var previousNames: [String: Int] = [:]
        var emptyCount = 0
        for child in m.children {
            let uniqueLabel: String
            if let label = child.label {
                if let nameCount = previousNames[label] {
                    uniqueLabel = "\(label)_\(nameCount)"
                    previousNames[label] = nameCount + 1
                } else {
                    uniqueLabel = label
                    previousNames[label] = 1
                }
            } else {
                uniqueLabel = "[\(emptyCount)]"
                emptyCount += 1
            }
            let path = (scope != nil ? scope! + separator : "") + uniqueLabel
            let compoundTypeDescription = "\(currentType).\(uniqueLabel)"
            if ignoredTensorPaths.contains(compoundTypeDescription) {
                continue
            }
            if let tensor = child.value as? Tensor<Float> {
                if tensors[path] != nil {
                    print(
                        "Warning: Saved two different tensors with the same name: \(path). This is most likely undesired behavior.")
                }
                tensors[path] = tensor
            } else {
                recursivelyObtainTensors(
                    child.value, scope: path, tensors: &tensors, separator: separator,
                    ignoredTensorPaths: ignoredTensorPaths)
            }
        }
    }
    
    static func remapTensorNames(
        tensors: [String: Tensor<Float>], nameMap: (String) -> String
    ) -> [String: Tensor<Float>] {
        var remappedTensors: [String: Tensor<Float>] = [:]
        for (key, value) in tensors {
            remappedTensors[nameMap(key)] = value
        }
        return remappedTensors
    }
    
    static func lookupMap(table: [String: String]) -> (String) -> String {
        return {name in
            return table[name] ?? name
        }
    }
    
    static func identityMap(_ name: String) -> String {
        return name
    }
}
