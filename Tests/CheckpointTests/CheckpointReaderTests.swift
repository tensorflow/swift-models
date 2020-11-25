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

@testable import Checkpoints

final class CheckpointReaderTests: XCTestCase {
    let resourceBaseLocation = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("SavedModels")

    func testLocalSavedModel() {
        // The data shard for the SavedModel has been replaced with a 0-byte placeholder. This
        // minimizes the size impact to the repository, while preserving the remaining structure.
        do {
            let reader = try CheckpointReader(
                checkpointLocation: resourceBaseLocation.appendingPathComponent("resnet.tar.gz"),
                modelName: "LocalTestCase", additionalFiles: [])

            XCTAssertEqual(reader.tensorCount, 252)
        } catch {
            XCTFail("Local SavedModel reading failed with error: \(error).")
        }
    }

    func testRemoteCheckpoint() {
        let mobilenetCheckpoint =
            URL(string: "http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_128.tgz")!

        do {
            let reader = try CheckpointReader(
                checkpointLocation: mobilenetCheckpoint,
                modelName: "RemoteTestCase", additionalFiles: [])

            XCTAssertEqual(reader.tensorCount, 443)
        } catch {
            XCTFail("Remote checkpoint reading and unarchiving failed with error: \(error).")
        }
    }
}

extension CheckpointReaderTests {
    static var allTests = [
        ("testLocalSavedModel", testLocalSavedModel),
        ("testRemoteCheckpoint", testRemoteCheckpoint),
    ]
}
