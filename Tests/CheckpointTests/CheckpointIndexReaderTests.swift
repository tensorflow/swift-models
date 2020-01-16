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

final class CheckpointIndexReaderTests: XCTestCase {
    let resourceBaseLocation = URL(fileURLWithPath: #file).deletingLastPathComponent()
        .appendingPathComponent("IndexFiles")

    func testLoadingMiniGo() {
        // TODO: Enable this test once either Snappy-compressed checkpoints are supported.
        // do {
        //     let indexReader = try CheckpointIndexReader(
        //         file: resourceBaseLocation.appendingPathComponent("minigo.index"))
        //     let header = try indexReader.readHeader()
        //     let keysAndValues = try indexReader.readAllKeysAndValues()

        //     XCTAssertEqual(header.numShards, 1)
        //     XCTAssertEqual(keysAndValues.count, 333)

        //     let firstTensorMetadata = keysAndValues["batch_normalization/beta"]
        //     XCTAssertNotNil(
        //         firstTensorMetadata, "batch_normalization/beta tensor missing from index file.")
        //     let secondTensorMetadata = keysAndValues["batch_normalization/beta/Momentum"]
        //     XCTAssertNotNil(
        //         secondTensorMetadata,
        //         "batch_normalization/beta/Momentum tensor missing from index file.")
        //     let lastTensorMetadata = keysAndValues["global_step"]
        //     XCTAssertNotNil(lastTensorMetadata, "global_step tensor missing from index file.")

        // } catch {
        //     XCTFail("MiniGo checkpoint index reading failed with error: \(error).")
        // }
    }

    func testLoadingTransformer() {
        do {
            let indexReader = try CheckpointIndexReader(
                file: resourceBaseLocation.appendingPathComponent("transformer.index"))
            let header = try indexReader.readHeader()
            let keysAndValues = try indexReader.readAllKeysAndValues()

            XCTAssertEqual(header.numShards, 1)
            XCTAssertEqual(keysAndValues.count, 148)

            let firstTensorMetadata = keysAndValues["model/h0/attn/c_attn/b"]
            XCTAssertNotNil(
                firstTensorMetadata, "model/h0/attn/c_attn/b tensor missing from index file.")
            let secondTensorMetadata = keysAndValues["model/h0/attn/c_proj/b"]
            XCTAssertNotNil(
                secondTensorMetadata, "model/h0/attn/c_proj/b tensor missing from index file.")
            let lastTensorMetadata = keysAndValues["model/wte"]
            XCTAssertNotNil(lastTensorMetadata, "model/wte tensor missing from index file.")

        } catch {
            XCTFail("MiniGo checkpoint index reading failed with error: \(error).")
        }
    }

    func testLoadingStyleTransfer() {
        do {
            let indexReader = try CheckpointIndexReader(
                file: resourceBaseLocation.appendingPathComponent("style.index"))
            let header = try indexReader.readHeader()
            let keysAndValues = try indexReader.readAllKeysAndValues()

            XCTAssertEqual(header.numShards, 1)
            XCTAssertEqual(keysAndValues.count, 92)

            let firstTensorMetadata = keysAndValues["conv1.conv2d.bias"]
            XCTAssertNotNil(
                firstTensorMetadata, "conv1.conv2d.bias tensor missing from index file.")
            let secondTensorMetadata = keysAndValues["conv1.conv2d.weight"]
            XCTAssertNotNil(
                secondTensorMetadata, "conv1.conv2d.weight tensor missing from index file.")
            let lastTensorMetadata = keysAndValues["res5.in2.weight"]
            XCTAssertNotNil(lastTensorMetadata, "res5.in2.weight tensor missing from index file.")

        } catch {
            XCTFail("MiniGo checkpoint index reading failed with error: \(error).")
        }
    }
}

extension CheckpointIndexReaderTests {
    static var allTests = [
        ("testLoadingMiniGo", testLoadingMiniGo),
        ("testLoadingTransformer", testLoadingTransformer),
        ("testLoadingStyleTransfer", testLoadingStyleTransfer),
    ]
}
