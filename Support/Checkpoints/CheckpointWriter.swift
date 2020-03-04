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

    /// Initializes the checkpoint reader from a dictionary of tensors, keyed on their string names.
    ///
    /// - Parameters:
    ///   - tensors: A dictionary containing the tensors to be written, with the keys being the
    ///     names of those tensors to write in the checkpoint.
    public init(tensors: [String: Tensor<Float>]) {
        self.tensors = tensors
    }

    /// Writes the checkpoint to disk, in a specified directory. A TensorFlow v2 checkpoint consists
    /// of a directory that contains a [name].index header file and one or more
    /// [name].data-0000X-of-0000Y binary shard files with the tensor bytes within them.
    ///
    /// - Parameters:
    ///   - directory: The directory to write the checkpoint into. If it doesn't exist, it will be
    ///     created.
    ///   - name: The base name of the checkpoint, which is what will have the .index and
    ///     .data-0000X-of-0000Y extensions appended to it for files in the checkpoint directory.
    public func write(to directory: URL, name: String) throws {
        try createDirectoryIfMissing(at: directory.path)
        let indexWriter = CheckpointIndexWriter(tensors: tensors)
        let indexHeader = indexWriter.serializedHeader()
        let headerLocation = directory.appendingPathComponent("\(name).index")
        try indexHeader.write(to: headerLocation)

        // TODO: Handle splitting into multiple shards.
        try writeShard(
            to: directory.appendingPathComponent("\(name)"), shard: 0, numShards: 1,
            tensorList: indexWriter.orderedTensors)
    }

    func writeShard(to location: URL, shard: Int, numShards: Int, tensorList: [String]) throws {
        let shardFile = CheckpointReader.shardFile(
            location: location, shard: shard, totalShards: numShards)

        var outputBuffer = Data()
        // TODO: Write this directly to disk, rather than accumulating it in memory.
        for tensorName in tensorList {
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

        try outputBuffer.write(to: shardFile)
    }
}
