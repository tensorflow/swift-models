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
import TensorFlow

/// A Swift-native TensorFlow v2 checkpoint writer. This writer has no dependencies
/// on the TensorFlow runtime or libraries.
open class CheckpointWriter {
    // TODO: Extend handling to different tensor types.
    let tensors: [String: Tensor<Float>]

    public init(tensors: [String: Tensor<Float>]) {
        self.tensors = tensors
    }

    public func write(to directory: URL, name: String) throws {
        try createDirectoryIfMissing(at: directory.path)
        let indexWriter = CheckpointIndexWriter(tensors: tensors)
        let indexHeader = indexWriter.serializedHeader()
        let headerLocation = directory.appendingPathComponent("\(name).index")
        try indexHeader.write(to: headerLocation)

        // Write shards
    }
}
