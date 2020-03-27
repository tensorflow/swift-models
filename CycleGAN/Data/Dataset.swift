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
import ModelSupport
import TensorFlow

public class Images {
    struct Elements: TensorGroup {
        var image: Tensor<Float>
    }

    let dataset: Dataset<Elements>
    let count: Int

    public init(folderURL: URL) throws {
        let folderContents = try FileManager.default
                                            .contentsOfDirectory(at: folderURL,
                                                                 includingPropertiesForKeys: [.isDirectoryKey],
                                                                 options: [.skipsHiddenFiles])
        let imageFiles = folderContents.filter { $0.pathExtension == "jpg" }

        var sourceData: [Float] = []

        var elements = 0

        for imageFile in imageFiles {
            let imageTensor = Image(jpeg: imageFile).tensor

            sourceData.append(contentsOf: imageTensor.scalars)

            elements += 1
        }

        let source = Tensor<Float>(shape: [elements, 256, 256, 3], scalars: sourceData) / 127.5 - 1.0
        dataset = Dataset(elements: Elements(image: source))
        count = elements
    }
}
