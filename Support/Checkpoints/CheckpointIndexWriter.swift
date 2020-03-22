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
    let orderedTensors: [String]

    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/io/table_options.h#L48
    let blockRestartInterval = 15

    init(tensors: [String: Tensor<Float>]) {
        self.tensors = tensors
        self.orderedTensors = tensors.keys.sorted()
    }
}

extension CheckpointIndexWriter {
    func serializedHeader() -> Data {
        var outputBuffer = Data()
        // TODO: Expand beyond using a single binary shard.
        outputBuffer.append(headerBlock(shards: 1))
        var lastString = ""
        var intervalSinceLastRestart = 1
        var restarts: [UInt32] = [0]
        var offset: Int64 = 0
        for key in orderedTensors {
            outputBuffer.append(keyValueBlock(key: key, lastString: lastString, offset: &offset))

            // With prefix compression, the entire string is used as a restart at defined intervals.
            if intervalSinceLastRestart < blockRestartInterval {
                lastString = key
                intervalSinceLastRestart += 1
            } else {
                lastString = ""
                intervalSinceLastRestart = 0
                restarts.append(UInt32(outputBuffer.count))
            }
        }

        restarts.append(UInt32(restarts.count))
        // Write the restart offsets as a trailer to the data block.
        restarts.withUnsafeBufferPointer { (ptr) in
            ptr.baseAddress!.withMemoryRebound(
                to: UInt8.self, capacity: ptr.count * MemoryLayout<UInt32>.size
            ) {
                outputBuffer.append($0, count: ptr.count * MemoryLayout<UInt32>.size)
            }
        }

        // The type of the block, with 0 signifying uncompressed.
        outputBuffer.append(contentsOf: [0])

        let crc32C = outputBuffer.maskedCRC32C()
        outputBuffer.append(contentsOf: crc32C.littleEndianBuffer)
        let headerSize = outputBuffer.count - 5

        // Data block is finished, terminate the file with meta index, index, and footer blocks.
        let metaIndex = metaIndexBlock()
        let metaIndexSize = metaIndex.count - 5
        let metaIndexOffset = outputBuffer.count
        outputBuffer.append(metaIndex)

        let index = indexBlock(lastKey: lastString, headerSize: headerSize)
        let indexSize = index.count - 5
        let indexOffset = outputBuffer.count
        outputBuffer.append(index)

        outputBuffer.append(
            footerBlock(
                metaIndexHandle: (metaIndexOffset, metaIndexSize),
                indexHandle: (indexOffset, indexSize)))

        return outputBuffer
    }

    // Based on the LevelDB implementation of the same function.
    func findShortestSuccessor(_ key: String) -> [UInt8] {
        var newKeyBytes: [UInt8] = []
        for byte in key.utf8 {
            let castByte = UInt8(byte)
            if castByte != 0xFF {
                newKeyBytes.append(castByte + 1)
                return newKeyBytes
            }
            newKeyBytes.append(castByte)
        }
        return newKeyBytes
    }

    func headerBlock(shards: Int) -> Data {
        var headerVersion = Tensorflow_VersionDef()
        headerVersion.producer = 1
        var headerProtobuf = Tensorflow_BundleHeaderProto()
        headerProtobuf.numShards = Int32(shards)
        headerProtobuf.version = headerVersion
        do {
            let headerValue = try headerProtobuf.serializedData()

            var outputBuffer = indexBytes(
                sharedKeyBytes: 0, newKeyBytes: 0, valueLength: headerValue.count)
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

        let tensorSize: Int64 = Int64(
            MemoryLayout<Float>.size * tensor.shape.dimensions.reduce(1) { $0 * $1 })

        entryProtobuf.dtype = .dtFloat
        entryProtobuf.shape = shape
        entryProtobuf.offset = offset
        entryProtobuf.size = tensorSize

        // TODO: Find a more efficient way of calculating this without casting to bytes twice.
        let scalars = tensor.array.scalars
        scalars.withUnsafeBufferPointer { (ptr) in
            ptr.baseAddress!.withMemoryRebound(
                to: UInt8.self, capacity: ptr.count * MemoryLayout<Float>.size
            ) {
                let tensorData = Data(bytes: $0, count: ptr.count * MemoryLayout<Float>.size)
                entryProtobuf.crc32C = tensorData.maskedCRC32C()
            }
        }

        offset += tensorSize

        do {
            let entryValue = try entryProtobuf.serializedData()
            let commonPrefix = lastString.commonPrefix(with: key)
            let newCharacters = key.count - commonPrefix.count
            var outputBuffer = indexBytes(
                sharedKeyBytes: commonPrefix.count, newKeyBytes: newCharacters,
                valueLength: entryValue.count)
            let suffix = key.suffix(newCharacters).utf8
            outputBuffer.append(contentsOf: suffix)
            outputBuffer.append(entryValue)
            return outputBuffer
        } catch {
            fatalError("Could not serialize header protobuf: \(error).")
        }
    }

    func indexBlock(lastKey: String, headerSize: Int) -> Data {
        var headerHandle = Data()
        headerHandle.appendVarint(0)
        headerHandle.appendVarint(headerSize)

        let shortestSuccessor = findShortestSuccessor(lastKey)
        var outputBuffer = indexBytes(
            sharedKeyBytes: 0, newKeyBytes: shortestSuccessor.count,
            valueLength: headerHandle.count)
        outputBuffer.append(contentsOf: shortestSuccessor)
        outputBuffer.append(headerHandle)

        // There were no restarts, but need to write out the buffer and count.
        outputBuffer.append(contentsOf: [0, 0, 0, 0])
        outputBuffer.append(contentsOf: [1, 0, 0, 0])

        // The type of the block, with 0 signifying uncompressed.
        outputBuffer.append(contentsOf: [0])

        let crc32C = outputBuffer.maskedCRC32C()
        outputBuffer.append(contentsOf: crc32C.littleEndianBuffer)

        return outputBuffer
    }

    func metaIndexBlock() -> Data {
        // The meta index block is unused, but is still written.
        var outputBuffer = Data()
        outputBuffer.append(contentsOf: [0, 0, 0, 0])
        outputBuffer.append(contentsOf: [1, 0, 0, 0, 0])

        let crc32C = outputBuffer.maskedCRC32C()
        outputBuffer.append(contentsOf: crc32C.littleEndianBuffer)

        return outputBuffer
    }

    func footerBlock(metaIndexHandle: (Int, Int), indexHandle: (Int, Int)) -> Data {
        // Footer format, as defined in LevelDB:
        // https://github.com/google/leveldb/blob/master/doc/table_format.md
        // Two handles (offset, size varint pairs) for the meta index and index blocks are followed
        // by zeroes to pad out to `footerSize - 8` bytes, with an 8-byte terminating magic number.
        var footerBytes = Data()
        footerBytes.appendVarint(metaIndexHandle.0)
        footerBytes.appendVarint(metaIndexHandle.1)
        footerBytes.appendVarint(indexHandle.0)
        footerBytes.appendVarint(indexHandle.1)

        footerBytes.append(Data(count: footerSize - 8 - footerBytes.count))
        let magicNumber: [UInt8] = [0x57, 0xFB, 0x80, 0x8B, 0x24, 0x75, 0x47, 0xDB]
        footerBytes.append(contentsOf: magicNumber)
        return footerBytes
    }

    func indexBytes(sharedKeyBytes: Int, newKeyBytes: Int, valueLength: Int) -> Data {
        var outputBuffer = Data()
        outputBuffer.appendVarint(sharedKeyBytes)
        outputBuffer.appendVarint(newKeyBytes)
        outputBuffer.appendVarint(valueLength)
        return outputBuffer
    }
}

extension Data {
    // Logic from https://github.com/apple/swift-protobuf/blob/master/Sources/SwiftProtobuf/BinaryEncoder.swift#L68
    mutating func appendVarint(_ value: Int) {
        var v = value
        while v > 127 {
            self.append(contentsOf: [UInt8(v & 0x7f | 0x80)])
            v >>= 7
        }
        self.append(contentsOf: [UInt8(v)])
    }
}

extension UInt32 {
    var littleEndianBuffer: [UInt8] {
        return [self].withUnsafeBufferPointer { (ptr) in
            ptr.baseAddress!.withMemoryRebound(
                to: UInt8.self, capacity: 4
            ) { [UInt8](UnsafeBufferPointer(start: $0, count: 4)) }
        }
    }
}
