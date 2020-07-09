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

import ModelSupport
import Foundation
import TensorFlow

/// Models that comply to Checkpointable can have their Tensors be written to and read from disk
/// using the `writeCheckpoint(to:...)` and `readCheckpoint(from:...)` interfaces.
public protocol Checkpointable: KeyPathIterable {
  /// Any Tensor that should be ignored for checkpoint reading or writing, specified in
  /// `Type.property` syntax. For example, `["Attention.scale"]`.
  var ignoredTensorPaths: Set<String> { get }
  
  /// The string separator between descending levels of the model. For example, a separator of `"/"`
  /// will yield tensor path names like `conv2/filter`.
  var checkpointSeparator: String { get }

  /// A mapping function between the internally generated tensor path names and how those names
  /// will or do appear in the on-disk checkpoint.
  var tensorNameMap: ((String) -> String) { get }
}

public extension Checkpointable {
  var ignoredTensorPaths: Set<String> {
    return []
  }
  
  var checkpointSeparator: String {
    return "/"
  }
  
  var tensorNameMap: (String) -> String {
    return CheckpointWriter.identityMap
  }

  /// Writes a checkpoint of this model's tensors to disk.
  ///
  /// - Parameters:
  ///   - location: The directory to write the checkpoint into. If it doesn't exist, it will be
  ///     created.
  ///   - name: The base name of the checkpoint, which is what will have the .index and
  ///     .data-0000X-of-0000Y extensions appended to it for files in the checkpoint directory.
  ///   - fileSystem: The filesystem used for writing the checkpoint. Defaults to
  ///     FoundationFileSystem.
  ///   - nameTable: A lookup table of generated tensor path names to their corresponding tensor
  ///     name in the checkpoint file. If an internal tensor path name is not represented, the
  ///     internal path name is used for the on-disk checkpoint.
  func writeCheckpoint(
    to location: URL, name: String, fileSystem: FileSystem = FoundationFileSystem(),
    nameTable: [String: String]
  ) throws {
    try writeCheckpoint(
      to: location, name: name, fileSystem: fileSystem,
      nameMap: CheckpointWriter.lookupMap(table: nameTable))
  }
  
  /// Writes a checkpoint of this model's tensors to disk.
  ///
  /// - Parameters:
  ///   - location: The directory to write the checkpoint into. If it doesn't exist, it will be
  ///     created.
  ///   - name: The base name of the checkpoint, which is what will have the .index and
  ///     .data-0000X-of-0000Y extensions appended to it for files in the checkpoint directory.
  ///   - fileSystem: The filesystem used for writing the checkpoint. Defaults to
  ///     FoundationFileSystem.
  ///   - nameMap: A mapping function that converts generated tensor path names to their
  ///     corresponding tensor name in the checkpoint file.
  func writeCheckpoint(
    to location: URL, name: String, fileSystem: FileSystem = FoundationFileSystem(),
    nameMap: ((String) -> String)? = nil
  ) throws {
    var rawTensors: [String: Tensor<Float>] = [:]
    CheckpointWriter.recursivelyObtainTensors(
      self, tensors: &rawTensors, separator: self.checkpointSeparator,
      ignoredTensorPaths: self.ignoredTensorPaths)
    
    let tensors = CheckpointWriter.remapTensorNames(tensors: rawTensors,
      nameMap: nameMap ?? self.tensorNameMap)
    
    let writer = CheckpointWriter(fileSystem: fileSystem)
    try writer.write(tensors: tensors, to: location, name: name)
  }
  
  /// Reads a checkpoint of this model's tensors from disk.
  ///
  /// - Parameters:
  ///   - location: Either a URL to the checkpoint files, where the last component is the file
  ///     base of the checkpoint files, or a URL to an archive containing the checkpoint files.
  ///   - name: The base name of the checkpoint, which is what will have the .index and
  ///     .data-0000X-of-0000Y extensions appended to it for files in the checkpoint directory.
  ///   - fileSystem: The filesystem used for reading the checkpoint. Defaults to
  ///     FoundationFileSystem.
  ///   - nameMap: A mapping function that converts generated tensor path names to their
  ///     corresponding tensor name in the checkpoint file.
  mutating func readCheckpoint(
    from location: URL, name: String, fileSystem: FileSystem = FoundationFileSystem(),
    nameMap: ((String) -> String)? = nil
  ) throws {
    var rawTensorNames: [String] = []
    CheckpointReader.recursivelyObtainTensorNames(
      self, tensors: &rawTensorNames, separator: self.checkpointSeparator,
      ignoredTensorPaths: self.ignoredTensorPaths)

    let concreteNameMap = nameMap ?? self.tensorNameMap
    let tensorNames = rawTensorNames.map{ concreteNameMap($0) }
    
    let keypaths = self.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self)
    
    guard keypaths.count == tensorNames.count else {
      fatalError(
        "The number of writable key paths: \(keypaths.count) did not match the number of tensor names: \(tensorNames.count)")
    }
    
    let reader: CheckpointReader = try CheckpointReader(checkpointLocation: location,
      modelName: name)

    for (index, keypath) in keypaths.enumerated() {
      self[keyPath: keypath] = Tensor<Float>(reader.loadTensor(named: tensorNames[index]))
    }
  }
  
  /// Reads a checkpoint of this model's tensors from disk.
  ///
  /// - Parameters:
  ///   - location: Either a URL to the checkpoint files, where the last component is the file
  ///     base of the checkpoint files, or a URL to an archive containing the checkpoint files.
  ///   - name: The base name of the checkpoint, which is what will have the .index and
  ///     .data-0000X-of-0000Y extensions appended to it for files in the checkpoint directory.
  ///   - fileSystem: The filesystem used for reading the checkpoint. Defaults to
  ///     FoundationFileSystem.
  ///   - nameTable: A lookup table of generated tensor path names to their corresponding tensor
  ///     name in the checkpoint file. If an internal tensor path name is not represented, the
  ///     internal path name is used for the on-disk checkpoint.
  mutating func readCheckpoint(
    from location: URL, name: String, fileSystem: FileSystem = FoundationFileSystem(),
    nameTable: [String: String]
  ) throws {
    try readCheckpoint(
      from: location, name: name, fileSystem: fileSystem,
      nameMap: CheckpointWriter.lookupMap(table: nameTable))
  }
}
