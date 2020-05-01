// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

func fetchMNISTDataset(
    localStorageDirectory: URL,
    remoteBaseDirectory: String,
    imagesFilename: String,
    labelsFilename: String,
    flattening: Bool,
    normalizing: Bool
) -> [TensorPair<Float, Int32>] {
    guard let remoteRoot = URL(string: remoteBaseDirectory) else {
        fatalError("Failed to create MNIST root url: \(remoteBaseDirectory)")
    }

    let imagesData = DatasetUtilities.fetchResource(
        filename: imagesFilename,
        fileExtension: "gz",
        remoteRoot: remoteRoot,
        localStorageDirectory: localStorageDirectory)
    let labelsData = DatasetUtilities.fetchResource(
        filename: labelsFilename,
        fileExtension: "gz",
        remoteRoot: remoteRoot,
        localStorageDirectory: localStorageDirectory)

    let images = [UInt8](imagesData).dropFirst(16).map(Float.init)
    let labels = [UInt8](labelsData).dropFirst(8).map(Int32.init)

    let rowCount = labels.count
    let (imageWidth, imageHeight) = (28, 28)

    if flattening {
        var flattenedImages =
            Tensor(shape: [rowCount, imageHeight * imageWidth], scalars: images)
            / 255.0
        if normalizing {
            flattenedImages = flattenedImages * 2.0 - 1.0
        }
        return (0..<rowCount).map {
            TensorPair(first: flattenedImages[$0], second: Tensor<Int32>(labels[$0]))
        }
    } else {
        var images =
            Tensor(shape: [rowCount, 1, imageHeight, imageWidth], scalars: images)
            .transposed(permutation: [0, 2, 3, 1]) / 255.0
        if normalizing {
            images = images * 2.0 - 1.0
        }
        return (0..<rowCount).map {
            TensorPair(first: images[$0], second: Tensor<Int32>(labels[$0]))
        }
    }
}
