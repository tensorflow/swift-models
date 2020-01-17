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

// This provides a simple decoder for Snappy-compressed data. Some TensorFlow v2 checkpoint index
// files are compressed with this, so we need a decoder for those.
//
// The Snappy compression format is described at https://github.com/google/snappy , specifically in
// https://github.com/google/snappy/blob/master/format_description.txt .

import Foundation

public extension Data {
    // Implementation derived from decodeVarint() in 
    // https://github.com/apple/swift-protobuf/blob/master/Sources/SwiftProtobuf/BinaryDecoder.swift
    func readVarint32(at index: inout Int) -> Int {
        let firstByte = self[index]
        index += 1
        if (firstByte & 0x80) == 0 {
            return Int(firstByte)
        }

        var value = Int(firstByte & 0x7f)
        var shift = 7

        while true {
            let currentByte = self[index]
            index += 1
            value |= Int(currentByte & 0x7f) << shift
            if currentByte & 0x80 == 0 {
                return value
            }
            shift += 7
        }
    }

    func decompressSnappyStream(at index: inout Int) -> Data? {
        let uncompressedLength = readVarint32(at: &index)
        print("Uncompressed length: \(uncompressedLength)")
        // Should start with 

        return nil
    }

    func decompressSnappy() -> Data {
        var decompressedData = Data()
        var index = 0

        while let value = decompressSnappyStream(at: &index) {
            decompressedData.append(value)
        }

        return decompressedData
    }
}
