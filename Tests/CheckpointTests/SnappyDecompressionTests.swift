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

import TensorFlow
import XCTest

@testable import ModelSupport

final class SnappyDecompressionTests: XCTestCase {
    let resourceBaseLocation = URL(fileURLWithPath: #file).deletingLastPathComponent()
        .appendingPathComponent("IndexFiles")
    
    func testReadingVarints() {
        let oneByteVarints = Data([24, 0, 63])
        var index = 0
        let varint1 = oneByteVarints.readVarint32(at: &index)
        XCTAssertEqual(index, 1)
        XCTAssertEqual(varint1, 24)
        let varint2 = oneByteVarints.readVarint32(at: &index)
        XCTAssertEqual(index, 2)
        XCTAssertEqual(varint2, 0)
        let varint3 = oneByteVarints.readVarint32(at: &index)
        XCTAssertEqual(index, 3)
        XCTAssertEqual(varint3, 63)
        
        let multiByteVarints = Data([172, 2])
        index = 0
        let varint4 = multiByteVarints.readVarint32(at: &index)
        XCTAssertEqual(index, 2)
        XCTAssertEqual(varint4, 300)
    }

    func testReadingData() {
        let sourceData = Data([23, 1, 5, 2, 7, 56])
        var index = 1
        let byte1 = sourceData.readByte(at: &index)
        XCTAssertEqual(index, 2)
        XCTAssertEqual(byte1, 1)
        let byte2 = sourceData.readByte(at: &index)
        XCTAssertEqual(index, 3)
        XCTAssertEqual(byte2, 5)
        let byteBuffer = Data(sourceData.readDataBlock(at: &index, size: 2))
        XCTAssertEqual(index, 5)
        XCTAssertEqual(byteBuffer[0], 2)
        XCTAssertEqual(byteBuffer[1], 7)
        let byte3 = sourceData.readByte(at: &index)
        XCTAssertEqual(index, 6)
        XCTAssertEqual(byte3, 56)
    }

    func testDecodingSnappyStream() {
        let literalTagByte: UInt8 = 0b00010100
        let copyTagByte: UInt8 =  0b00000001

        // Construct a byte buffer that inlines 6 bytes and copies 4 of them afterward.
        let snappyData = Data([10, literalTagByte, 12, 1, 50, 3, 45, 13, copyTagByte, 4])
        do {
            let uncompressedData = try snappyData.decompressFromSnappy()
            XCTAssertEqual(uncompressedData.count, 10)
            XCTAssertEqual(uncompressedData[0], 12)
            XCTAssertEqual(uncompressedData[6], 50)
        } catch {
            XCTFail("Decompressing sample data failed with error \(error).")
        }
    }

    func testDecompressingMiniGo() {
        do {
            let miniGoData = try Data(
                contentsOf: resourceBaseLocation.appendingPathComponent("minigo.index"))
            let decompressedMiniGoData = try miniGoData.decompressFromSnappy()
            XCTAssertEqual(decompressedMiniGoData[0], 0)
        } catch {
            XCTFail("Decompressing MiniGo index failed with error \(error).")
        }
    }
}

extension SnappyDecompressionTests {
    static var allTests = [
        ("testReadingVarints", testReadingVarints),
        ("testReadingData", testReadingData),
        ("testDecodingSnappystream", testDecodingSnappyStream),
        ("testDecompressingMiniGo", testDecompressingMiniGo),
    ]
}
