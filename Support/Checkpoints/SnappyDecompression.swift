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

public enum SnappyDecompressionError: Error {
    case illegalLiteralLength(upperBits: UInt8)
    case impossibleTagType(tagType: UInt8)
}

// The following extension to Data provides methods that read variable-length byte sequences
// starting at an incoming index, then mutate the index by advancing it to the next read position.
public extension Data {
    // Implementation derived from decodeVarint() in 
    // https://github.com/apple/swift-protobuf/blob/master/Sources/SwiftProtobuf/BinaryDecoder.swift
    func readVarint32(at index: inout Int) -> Int {
        let firstByte = readByte(at: &index)
        if (firstByte & 0x80) == 0 {
            return Int(firstByte)
        }

        var value = Int(firstByte & 0x7f)
        var shift = 7

        while true {
            let currentByte = readByte(at: &index)
            value |= Int(currentByte & 0x7f) << shift
            if currentByte & 0x80 == 0 {
                return value
            }
            shift += 7
        }
    }

    func readByte(at index: inout Int) -> UInt8 {
        let byte =  self[index]
        index += 1
        return byte
    }

    func readDataBlock(at index: inout Int, size: Int) -> Data {
        let dataBlock = self[index..<(index + size)]
        index += size
        return dataBlock
    }

    func decompressSnappyStream(at index: inout Int) throws -> Data? {
        guard index < self.count else { return nil }
        
        let uncompressedLength = readVarint32(at: &index)

        var uncompressedData = Data()
        while uncompressedData.count < uncompressedLength {
            // Each section starts with a tag byte, which determines whether to read a sequence of
            // bytes directly into the uncompressed data (literal) or to copy a sequence of
            // previously-decompressed bytes into this position. The last two bits indicate the
            // class of the section, and the remaining bits encode class-specific information like
            // how many offset or length bytes follow or the length of the section to copy.
            let tagByte = readByte(at: &index)
            let tagType = tagByte & 0b00000011
            let upperBits = tagByte >> 2
            switch tagType {
            case 0: // Literal string of bytes.
                let literalLength: Int
                switch upperBits {
                case 0..<60: // Literal length is encoded in the upper bits of the tag byte.
                    literalLength = Int(upperBits) + 1
                case 60: // One-byte literal length following the tag byte.
                    literalLength = Int(readByte(at: &index)) + 1
                case 61: // Two-byte literal length following the tag byte.
                    let firstByte = readByte(at: &index)
                    let secondByte = readByte(at: &index)
                    literalLength = Int(firstByte) + Int(secondByte) * 256 + 1
                case 62: // Three-byte literal length following the tag byte.
                    let firstByte = readByte(at: &index)
                    let secondByte = readByte(at: &index)
                    let thirdByte = readByte(at: &index)
                    literalLength = Int(firstByte) + Int(secondByte) * 256 + Int(thirdByte) * 256
                        * 256 + 1
                case 63: // Four-byte literal length following the tag byte.
                    let firstByte = readByte(at: &index)
                    let secondByte = readByte(at: &index)
                    let thirdByte = readByte(at: &index)
                    let fourthByte = readByte(at: &index)
                    literalLength = Int(firstByte) + Int(secondByte) * 256 + Int(thirdByte) * 256
                        * 256 + Int(fourthByte) * 256 * 256 * 256 + 1
                default:
                    throw SnappyDecompressionError.illegalLiteralLength(upperBits: upperBits)
                }
                let literalData = self.readDataBlock(at: &index, size: literalLength)
                uncompressedData.append(literalData)
            case 1: // Copy with 1-byte offset.
                let copyLength = Int(upperBits & 0b00000111) + 4
                let upperOffset = (upperBits & 0b00111000) >> 3
                let lowerOffset = readByte(at: &index)
                
                let offset = Int(upperOffset) * 256 + Int(lowerOffset)
                var sourceIndex = uncompressedData.count - offset
                if offset < copyLength {
                    // Perform run-length encoding for offsets that cause reading past the end of
                    // the file.
                    let copiedBytes = copyLength - offset
                    let copyData = uncompressedData.readDataBlock(at: &sourceIndex, size: offset)
                    uncompressedData.append(copyData)
                    sourceIndex = uncompressedData.count - offset
                    let additionalData = uncompressedData.readDataBlock(
                        at: &sourceIndex, size: copiedBytes)
                    uncompressedData.append(additionalData)
                } else {
                    let copyData = uncompressedData.readDataBlock(
                        at: &sourceIndex, size: copyLength)
                    uncompressedData.append(copyData)
                }
            case 2: // Copy with 2-byte offset.
                let copyLength = Int(upperBits) + 1
                let firstByte = readByte(at: &index)
                let secondByte = readByte(at: &index)
                var sourceIndex = uncompressedData.count - (Int(firstByte) + Int(secondByte) * 256)
                let copyData = uncompressedData.readDataBlock(at: &sourceIndex, size: copyLength)
                uncompressedData.append(copyData)
            case 3: // Copy with 4-byte offset.
                let copyLength = Int(upperBits) + 1
                let firstByte = readByte(at: &index)
                let secondByte = readByte(at: &index)
                let thirdByte = readByte(at: &index)
                let fourthByte = readByte(at: &index)
                var sourceIndex = uncompressedData.count - (Int(firstByte) + Int(secondByte) * 256
                    + Int(thirdByte) * 256 * 256 + Int(fourthByte) * 256 * 256 * 256)
                let copyData = uncompressedData.readDataBlock(at: &sourceIndex, size: copyLength)
                uncompressedData.append(copyData)
            default:
                throw SnappyDecompressionError.impossibleTagType(tagType: tagType)
            }
        }
        if uncompressedData.count != uncompressedLength {
            // TODO: Determine if this should be elevated to a thrown error.
            printError(
                "Warning: uncompressed data length of \(uncompressedData.count) did not match desired length of \(uncompressedLength).")
        }
        
        return uncompressedData
    }

    // This assumes a single compressed block at the start of the file, and an uncompressed footer.
    func decompressFromSnappy() throws -> Data {
        var decompressedData = Data()
        var index = 0

        if let value = try decompressSnappyStream(at: &index) {
            decompressedData.append(value)
        }
        
        if index < (self.count - 1) {
            let footer = readDataBlock(at: &index, size: self.count - index - 1)
            decompressedData.append(footer)
        }

        return decompressedData
    }
}
