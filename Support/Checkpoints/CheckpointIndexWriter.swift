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
        // TODO: Calculate the number of shards required.
        outputBuffer.append(headerBlock(shards: 1))
        // sort strings
        // write each string key
        // generate tensor protobuf
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