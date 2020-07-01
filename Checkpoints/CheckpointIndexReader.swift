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

// TensorFlow v2 checkpoints use an index file as a key-value store to map saved tensor names to
// the metadata for each tensor. The format of this file is defined by tensorflow::table::Table
//
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/io/table_format.txt
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/io/table.h
//
// and consists of a series of string keys and associated data values. It is based on the LevelDB 
// table format: https://github.com/google/leveldb
//
// The very first key is a null string and its value is a protobuf containing header information
// about the entire checkpoint bundle (number of shards, etc.). The remaining keys are
// prefix-compressed strings in ascending alphabetical order representing each named tensor in the
// checkpoint, with their values being protobufs that contain metadata about each tensor.
//
// The binary data representing the tensors are stored in one or more shard files, with lookup
// locations determined by this metadata.

import Foundation
import ModelSupport

// The block footer size is constant, and is obtained from the following:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/io/format.h
// `2 * BlockHandle::kMaxEncodedLength + 8` where `kMaxEncodedLength = 10 + 10`
let footerSize = 48

class CheckpointIndexReader {
    let binaryData: Data
    var index: Int = 0
    var currentPrefix = Data()

    var atEndOfFile: Bool { return index >= (binaryData.count - footerSize - 1) }

  init(file: URL, fileSystem: FileSystem) throws {
        let indexFile = fileSystem.open(file.path)
        let fileData = try indexFile.read()
        if fileData[0] == 0 {
            binaryData = fileData
        } else {
            binaryData = try fileData.decompressFromSnappy()
        }
    }

    func resetHead() {
        index = 0
    }
}

// The main interface for iterating through all metadata contained in the index file.
extension CheckpointIndexReader {
    func readHeader() throws -> Tensorflow_BundleHeaderProto {
        // The header has a string key of "", so there's nothing to read for the key.
        // If a non-zero initial value is encountered, the file is Snappy-compressed, so we bail out
        // until it can be uncompressed.
        let initialValue = binaryData.readVarint32(at: &index)
        guard initialValue == 0 else {
            fatalError("Snappy-compressed data should have been picked up earlier than this.")
        }
        let _ = binaryData.readVarint32(at: &index)
        let valueLength = binaryData.readVarint32(at: &index)
        let value = binaryData.readDataBlock(at: &index, size: valueLength)

        let tempHeader = try Tensorflow_BundleHeaderProto(serializedData: value)
        return tempHeader
    }

    func readAllKeysAndValues() throws -> [String: Tensorflow_BundleEntryProto] {
        var lookupTable: [String: Tensorflow_BundleEntryProto] = [:]
        while let (key, value) = try readKeyAndValue() {
            lookupTable[key] = value
        }

        return lookupTable
    }
}

// The internal file parsing methods for smaller datatypes that comprise the key-value groupings.
extension CheckpointIndexReader {
    func readKey(sharedBytes: Int, unsharedBytes: Int) -> String {
        let newBytes = binaryData.readDataBlock(at: &index, size: unsharedBytes)
        guard sharedBytes <= currentPrefix.count else {
            fatalError(
                "Shared bytes of \(sharedBytes) exceeded stored prefix size of \(currentPrefix.count)."
            )
        }
        let keyData = currentPrefix[0..<sharedBytes] + newBytes
        currentPrefix = keyData
        return String(bytes: keyData, encoding: .utf8)!
    }

    func readKeyAndValue() throws -> (String, Tensorflow_BundleEntryProto)? {
        guard !atEndOfFile else { return nil }

        let sharedKeyBytes = binaryData.readVarint32(at: &index)
        let unsharedKeyBytes = binaryData.readVarint32(at: &index)
        let valueLength = binaryData.readVarint32(at: &index)
        let key = readKey(sharedBytes: sharedKeyBytes, unsharedBytes: unsharedKeyBytes)
        let value = binaryData.readDataBlock(at: &index, size: valueLength)

        // This relies on the fact that the first reset is at index 0, leading to four zeroes at the
        // start of the reset block. We should be able to read the length of the data block from 
        // the footer, when it is complete, but this should be viable for known checkpoints.
        if (sharedKeyBytes + unsharedKeyBytes + valueLength) == 0 { return nil }

        let bundleEntry = try Tensorflow_BundleEntryProto(serializedData: value)

        return (key, bundleEntry)
    }
}
