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

import XCTest
import ModelSupport
import TensorFlow

final class ImageTests: XCTestCase {
    let resourceBaseLocation = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("Images")
    let temporaryDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(
        "TestImages", isDirectory: true)

    override func setUp() {
        super.setUp()
        do {
            try createDirectoryIfMissing(at: temporaryDirectory.path)
        } catch {
            XCTFail("Unable to create temporary directory with error: \(error).")
        }
        
        // Remove pre-existing test files.
        let firstImagePath = temporaryDirectory.appendingPathComponent("testimage.jpg").path
        if FileManager.default.fileExists(atPath: firstImagePath) {
            try! FileManager.default.removeItem(atPath: firstImagePath)
        }
        let secondImagePath = temporaryDirectory.appendingPathComponent("testimage2.jpg").path
        if FileManager.default.fileExists(atPath: secondImagePath) {
            try! FileManager.default.removeItem(atPath: secondImagePath)
        }
    }

    func testImageLoading() {
        let imageLocation = resourceBaseLocation.appendingPathComponent("testimage.jpg")
        let rgbImage = Image(contentsOf: imageLocation, byteOrdering: .rgb)
        let imageTensor = rgbImage.tensor
        XCTAssertEqual(imageTensor.shape, [20, 60, 3])
        // Note: JPEG compression artifacts lead to these values being slightly off pure colors.
        XCTAssertEqual(imageTensor[5][5][0], Tensor<Float>(254.0))
        XCTAssertEqual(imageTensor[5][5][1], Tensor<Float>(0.0))
        XCTAssertEqual(imageTensor[5][5][2], Tensor<Float>(0.0))
        XCTAssertEqual(imageTensor[5][25][0], Tensor<Float>(0.0))
        XCTAssertEqual(imageTensor[5][25][1], Tensor<Float>(255.0))
        XCTAssertEqual(imageTensor[5][25][2], Tensor<Float>(1.0))
        XCTAssertEqual(imageTensor[5][45][0], Tensor<Float>(0.0))
        XCTAssertEqual(imageTensor[5][45][1], Tensor<Float>(0.0))
        XCTAssertEqual(imageTensor[5][45][2], Tensor<Float>(254.0))
        XCTAssertEqual(imageTensor[15][5][0], Tensor<Float>(123.0))
        XCTAssertEqual(imageTensor[15][5][1], Tensor<Float>(128.0))
        XCTAssertEqual(imageTensor[15][5][2], Tensor<Float>(134.0))
        XCTAssertEqual(imageTensor[15][25][0], Tensor<Float>(4.0))
        XCTAssertEqual(imageTensor[15][25][1], Tensor<Float>(0.0))
        XCTAssertEqual(imageTensor[15][25][2], Tensor<Float>(0.0))
        XCTAssertEqual(imageTensor[15][45][0], Tensor<Float>(255.0))
        XCTAssertEqual(imageTensor[15][45][1], Tensor<Float>(255.0))
        XCTAssertEqual(imageTensor[15][45][2], Tensor<Float>(250.0))
    }
    
    func testImageSaving() {
        let imageDestination = temporaryDirectory.appendingPathComponent("testimage.jpg")

        let rgbTensor = Tensor<Float>(ones: [15, 20, 3])
        let rgbImage = Image(rgbTensor)
        rgbImage.save(to: imageDestination, format: .jpeg(quality: 95))
        let reloadedRGBImage = Image(contentsOf: imageDestination, byteOrdering: .rgb)
        XCTAssertEqual(reloadedRGBImage.tensor.shape, [15, 20, 3])

        let grayscaleTensor = Tensor<Float>(ones: [15, 20, 1])
        let grayscaleImage = Image(grayscaleTensor)
        grayscaleImage.save(to: imageDestination, format: .jpeg(quality: 95))
        let reloadedGrayscaleImage = Image(contentsOf: imageDestination, byteOrdering: .rgb)
        XCTAssertEqual(reloadedGrayscaleImage.tensor.shape, [15, 20, 3])
        
        let imageSource = resourceBaseLocation.appendingPathComponent("testimage.jpg")
        let imageDestination2 = temporaryDirectory.appendingPathComponent("testimage.jpg")
        let loadedRGBImage = Image(contentsOf: imageSource, byteOrdering: .rgb)
        loadedRGBImage.save(to: imageDestination2, format: .jpeg(quality: 95))
        let reloadedRGBImage2 = Image(contentsOf: imageDestination2, byteOrdering: .rgb)
        XCTAssertEqual(reloadedRGBImage2.tensor.shape, [20, 60, 3])
    }
    
    static var allTests = [
        ("testImageLoading", testImageLoading),
        ("testImageSaving", testImageSaving),
    ]
}
