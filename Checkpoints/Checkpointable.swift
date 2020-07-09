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

public protocol Checkpointable {
  var ignoredTensorPaths: Set<String> { get }
  var checkpointSeparator: String { get }
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

  func writeCheckpoint(
    to location: URL, name: String, fileSystem: FileSystem = FoundationFileSystem(),
    nameTable: [String: String]
  ) throws {
    try writeCheckpoint(
      to: location, name: name, fileSystem: fileSystem,
      nameMap: CheckpointWriter.lookupMap(table: nameTable))
  }
  
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
}
