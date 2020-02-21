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

class CheckpointIndexWriter {
    // TODO: Extend handling to different tensor types.
    let tensors: [String: Tensor<Float>]

    init(tensors: [String: Tensor<Float>]) {
        self.tensors = tensors
    }
}

extension CheckpointIndexWriter {
    func serializedHeader() -> Data {
        var outputBuffer = Data()
        // TODO: Expand beyond using a single binary shard.
        outputBuffer.append(headerBlock(shards: 1))
        let sortedKeys = tensors.keys.sorted()
        var lastString = ""
        var offset: Int64 = 0
        for key in sortedKeys {
            outputBuffer.append(keyValueBlock(key: key, lastString: lastString, offset: &offset))
            lastString = key
        }

        // TODO: Complete footer output, rather than just using these terminating zeroes.
        outputBuffer.append(contentsOf: [0, 0, 0])
        return outputBuffer
    }

    func headerBlock(shards: Int) -> Data {
        var headerVersion = Tensorflow_VersionDef()
        headerVersion.producer = 1
        var headerProtobuf = Tensorflow_BundleHeaderProto()
        headerProtobuf.numShards = Int32(shards)
        headerProtobuf.version = headerVersion
        do {
            let headerValue = try headerProtobuf.serializedData()

            var outputBuffer = indexBytes(sharedKeyBytes: 0, newKeyBytes: 0, valueLength: headerValue.count)
            outputBuffer.append(headerValue)

            return outputBuffer
        } catch {
            fatalError("Could not serialize header protobuf: \(error).")
        }
    }

    func keyValueBlock(key: String, lastString: String, offset: inout Int64) -> Data {
        guard let tensor = tensors[key] else { fatalError("Mismatch on tensor key: \(key).") }
        
        var entryProtobuf = Tensorflow_BundleEntryProto()
        var shape = Tensorflow_TensorShapeProto()
        shape.dim = tensor.shape.dimensions.map { size -> Tensorflow_TensorShapeProto.Dim in
            var dim = Tensorflow_TensorShapeProto.Dim()
            dim.size = Int64(size)
            return dim
        }
        
        let tensorSize: Int64 = Int64(MemoryLayout<Float>.size * tensor.shape.dimensions.reduce(1) { $0 * $1 })

        entryProtobuf.dtype = .dtFloat
        entryProtobuf.shape = shape
        entryProtobuf.offset = offset
        entryProtobuf.size = tensorSize
        entryProtobuf.crc32C = 10

        offset += tensorSize

        do {
            let entryValue = try entryProtobuf.serializedData()
            let commonPrefix = lastString.commonPrefix(with: key)
            let newCharacters = key.count - commonPrefix.count
            var outputBuffer = indexBytes(sharedKeyBytes: commonPrefix.count, newKeyBytes: newCharacters, valueLength: entryValue.count)
            let suffix = key.suffix(newCharacters).utf8
            outputBuffer.append(contentsOf: suffix)
            outputBuffer.append(entryValue)
            return outputBuffer
        } catch {
            fatalError("Could not serialize header protobuf: \(error).")
        }
    }

    func indexBytes(sharedKeyBytes: Int, newKeyBytes: Int, valueLength: Int) -> Data {
        var outputBuffer = Data()
        outputBuffer.appendVarint32(sharedKeyBytes)
        outputBuffer.appendVarint32(newKeyBytes)
        outputBuffer.appendVarint32(valueLength)
        return outputBuffer
    }
}

extension Data {
    mutating func appendVarint32(_ value: Int) {
        // TODO: Actual varint writing here
        self.append(contentsOf: [UInt8(value)])
    }
}
