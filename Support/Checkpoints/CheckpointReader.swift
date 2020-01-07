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

/// A shell for what will be a Swift-native checkpoint reader. Currently supplements the internal 
/// C API TensorFlow checkpoint reader with the ability to download remote checkpoints.
open class CheckpointReader {
    fileprivate let reader: TensorFlowCheckpointReader

    /// The local checkpoint location.
    public let localCheckpointLocation: URL

    /// The number of tensors stored in the checkpoint.
    public var tensorCount: Int { reader.tensorCount }

    /// The names of the tensors stored in the checkpoint.
    public var tensorNames: [String] { reader.tensorNames }

    /// Initializes the checkpoint reader from either a local or remote directory. If remote, 
    /// automatically downloads the checkpoint files into a temporary directory.
    ///
    /// - Parameters:
    ///   - checkpointLocation: A URL to the checkpoint files, where the last component is the file 
    ///     base of the checkpoint files.
    ///   - modelName: A distinct name for the model, to ensure that checkpoints with the same base 
    ///     name but for different models don't collide when downloaded.
    ///   - shards: The number of shards the weights have been split into. This is a temporary 
    ///     parameter until this can be read directly from the index.
    public init(
        checkpointLocation: URL, modelName: String, shards: Int = 1, additionalFiles: [String] = []
    ) {
        let checkpointBase = checkpointLocation.lastPathComponent
        if checkpointLocation.isFileURL {
            self.localCheckpointLocation = checkpointLocation
        } else {
            let temporaryDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(
                modelName)
            let temporaryCheckpointBase = temporaryDirectory.appendingPathComponent(checkpointBase)
            self.localCheckpointLocation = temporaryCheckpointBase
            if !FileManager.default.fileExists(atPath: temporaryDirectory.path) {
                do {
                    try CheckpointReader.downloadCheckpointFiles(
                        from: checkpointLocation, to: temporaryDirectory, shards: shards,
                        additionalFiles: additionalFiles)
                } catch {
                    fatalError("Failed to fetch and save checkpoint with error: \(error)")
                }
            }
        }

        self.reader = TensorFlowCheckpointReader(checkpointPath: self.localCheckpointLocation.path)
    }

    /// Constructs the file names for checkpoint components from a base URL and downloads them to a
    /// target directory.
    static func downloadCheckpointFiles(
        from checkpointLocation: URL, to temporaryDirectory: URL, shards: Int,
        additionalFiles: [String]
    ) throws {
        let indexFile = checkpointLocation.appendingPathExtension("index")
        try download(from: indexFile, to: temporaryDirectory)
        for shard in 0..<shards {
            let formatter = NumberFormatter()
            formatter.numberStyle = .decimal
            formatter.minimumIntegerDigits = 5
            formatter.maximumFractionDigits = 0
            formatter.hasThousandSeparators = false
            formatter.usesGroupingSeparator = false
            let currentShard = formatter.string(from: shard as NSNumber)!
            let totalShards = formatter.string(from: shards as NSNumber)!
            let shardFile = checkpointLocation.appendingPathExtension(
                "data-\(currentShard)-of-\(totalShards)"
            )
            try download(from: shardFile, to: temporaryDirectory)
        }
        let checkpointDirectory = checkpointLocation.deletingLastPathComponent()
        for file in additionalFiles {
            let additionalFile = checkpointDirectory.appendingPathComponent(file)
            try download(from: additionalFile, to: temporaryDirectory)
        }
    }

    /// Returns `true` if the checkpoint contains a tensor with the provided name.
    public func containsTensor(named name: String) -> Bool {
        return reader.containsTensor(named: name)
    }

    /// Returns the shape of the tensor with the provided name stored in the checkpoint.
    public func shapeOfTensor(named name: String) -> TensorShape {
        return reader.shapeOfTensor(named: name)
    }

    /// Returns the scalar type of the tensor with the provided name stored in the checkpoint.
    public func scalarTypeOfTensor(named name: String) -> Any.Type {
        return reader.scalarTypeOfTensor(named: name)
    }

    /// Loads and returns the value of the tensor with the provided name stored in the checkpoint.
    public func loadTensor<Scalar: _TensorFlowDataTypeCompatible>(
        named name: String
    ) -> ShapedArray<Scalar> {
        return reader.loadTensor(named: name)
    }
}
