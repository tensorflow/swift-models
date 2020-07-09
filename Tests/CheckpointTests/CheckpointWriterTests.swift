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
import ImageClassificationModels

extension LeNet: Checkpointable {}

extension SqueezeNetV1_0: Checkpointable {
    public var checkpointSeparator: String {
      return "_"
    }
}

final class CheckpointWriterTests: XCTestCase {
    let temporaryDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(
        "TestCase", isDirectory: true)

    override func setUp() {
        super.setUp()
        // Remove pre-existing test files.
        let headerPath = FileManager.default.temporaryDirectory.appendingPathComponent(
            "testmodel.ckpt.index"
        ).path
        if FileManager.default.fileExists(atPath: headerPath) {
            try! FileManager.default.removeItem(atPath: headerPath)
        }
        let shardPath = FileManager.default.temporaryDirectory.appendingPathComponent(
            "testmodel.ckpt.data-00000-of-00001"
        ).path
        if FileManager.default.fileExists(atPath: shardPath) {
            try! FileManager.default.removeItem(atPath: shardPath)
        }
    }

    func testCheckpointRoundtrip() {
        let vector = Tensor<Float>([1])
        let matrix = Tensor<Float>([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        let ones = Tensor<Float>(ones: [1, 2, 2, 2, 2, 2, 1])
        let tensor = Tensor<Float>(
            shape: [3, 4, 5], scalars: [Float](stride(from: 0.0, to: 60.0, by: 1.0)))

        let tensors = [
            "model/vector": vector, "model/matrix": matrix, "ones": ones, "tensor": tensor,
        ]

        do {
            let writer = CheckpointWriter()
            try writer.write(tensors: tensors, to: temporaryDirectory, name: "testmodel.ckpt")
            let reader = try CheckpointReader(
                checkpointLocation: temporaryDirectory.appendingPathComponent("testmodel.ckpt"),
                modelName: "TestCase", additionalFiles: [])

            let loadedVector: ShapedArray<Float> = reader.loadTensor(named: "model/vector")
            let loadedMatrix: ShapedArray<Float> = reader.loadTensor(named: "model/matrix")
            let loadedOnes: ShapedArray<Float> = reader.loadTensor(named: "ones")
            let loadedTensor: ShapedArray<Float> = reader.loadTensor(named: "tensor")
            XCTAssertEqual(loadedVector.shape, [1])
            XCTAssertEqual(loadedMatrix.shape, [2, 3])
            XCTAssertEqual(loadedOnes.shape, [1, 2, 2, 2, 2, 2, 1])
            XCTAssertEqual(loadedTensor.shape, [3, 4, 5])
            XCTAssertEqual(loadedTensor.scalars[3], 3.0)
        } catch {
            XCTFail("Checkpoint writing / reading failed with error: \(error).")
        }
    }
    
    func testLeNetCheckpointing() {
        do {
            let model = LeNet()
            try model.writeCheckpoint(to: temporaryDirectory, name: "LeNet")
            
            let reader = try CheckpointReader(
                checkpointLocation: temporaryDirectory.appendingPathComponent("LeNet"),
                modelName: "LeNet", additionalFiles: [])

            XCTAssertEqual(reader.tensorCount, 10)
            let loadedTensor1: ShapedArray<Float> = reader.loadTensor(named: "conv2/filter")
            XCTAssertEqual(loadedTensor1.shape, [5, 5, 6, 16])
            let loadedTensor2: ShapedArray<Float> = reader.loadTensor(named: "fc2/bias")
            XCTAssertEqual(loadedTensor2.shape, [84])
        } catch {
            XCTFail("LeNet checkpoint writing / reading failed with error: \(error).")
        }
    }

    func testSqueezeNetCheckpointing() {
        do {
            let model = SqueezeNetV1_0(classCount: 1000)
            try model.writeCheckpoint(to: temporaryDirectory, name: "SqueezeNet")
            
            let reader = try CheckpointReader(
                checkpointLocation: temporaryDirectory.appendingPathComponent("SqueezeNet"),
                modelName: "SqueezeNet", additionalFiles: [])

            XCTAssertEqual(reader.tensorCount, 52)
            let loadedTensor1: ShapedArray<Float> = reader.loadTensor(named: "fire2_expand3_filter")
            XCTAssertEqual(loadedTensor1.shape, [3, 3, 16, 64])
            let loadedTensor2: ShapedArray<Float> = reader.loadTensor(named: "conv10_filter")
            XCTAssertEqual(loadedTensor2.shape, [1, 1, 512, 1000])
        } catch {
            XCTFail("SqueezeNet checkpoint writing / reading failed with error: \(error).")
        }
    }
}

extension CheckpointWriterTests {
    static var allTests = [
        ("testCheckpointRoundtrip", testCheckpointRoundtrip),
        ("testLeNetCheckpointing", testLeNetCheckpointing),
        ("testSqueezeNetCheckpointing", testSqueezeNetCheckpointing),
    ]
}
