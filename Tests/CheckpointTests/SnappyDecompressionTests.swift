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
        ("testDecompressingMiniGo", testDecompressingMiniGo),
    ]
}
