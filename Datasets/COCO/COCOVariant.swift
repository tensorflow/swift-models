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

import Foundation
import ModelSupport

public struct COCOVariant {
    static let trainAnnotationsURL =
        URL(
            string:
                "https://storage.googleapis.com/s4tf-hosted-binaries/datasets/COCO/annotations-train2017.zip"
        )!
    static let valAnnotationsURL =
        URL(
            string:
                "https://storage.googleapis.com/s4tf-hosted-binaries/datasets/COCO/annotations-val2017.zip"
        )!
    static let testAnnotationsURL =
        URL(
            string:
                "https://storage.googleapis.com/s4tf-hosted-binaries/datasets/COCO/annotations-test2017.zip"
        )!
    static let testDevAnnotationsURL =
        URL(
            string:
                "https://storage.googleapis.com/s4tf-hosted-binaries/datasets/COCO/annotations-test-dev2017.zip"
        )!

    static let trainImagesURL =
        URL(string: "http://images.cocodataset.org/zips/train2017.zip")!
    static let valImagesURL =
        URL(string: "http://images.cocodataset.org/zips/val2017.zip")!
    static let testImagesURL =
        URL(string: "http://images.cocodataset.org/zips/test2017.zip")!

    static func downloadIfNotPresent(
        from location: URL,
        to directory: URL,
        filename: String
    ) {
        let downloadPath = directory.appendingPathComponent(filename).path
        let directoryExists = FileManager.default.fileExists(atPath: downloadPath)
        let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: downloadPath)
        let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)

        guard !directoryExists || directoryEmpty else { return }

        let _ = DatasetUtilities.downloadResource(
            filename: filename, fileExtension: "zip",
            remoteRoot: location.deletingLastPathComponent(), localStorageDirectory: directory)
    }

    static func loadJSON(_ directory: URL, annotations: String, images: String?) -> COCO {
        let jsonPath = directory.appendingPathComponent(annotations).path
        let jsonURL = URL(string: jsonPath)!
        var imagesDirectory: URL? = nil
        if images != nil {
            imagesDirectory = directory.appendingPathComponent(images!)
        }
        let coco = try! COCO(fromFile: jsonURL, imagesDirectory: imagesDirectory)
        return coco
    }

    public static func defaultDirectory() -> URL {
        return DatasetUtilities.defaultDirectory
            .appendingPathComponent("COCO", isDirectory: true)
    }

    public static func loadTrain(
        to directory: URL = defaultDirectory(),
        downloadImages: Bool = false
    ) -> COCO {
        downloadIfNotPresent(
            from: trainAnnotationsURL, to: directory,
            filename: "annotations-train2017")
        if downloadImages {
            downloadIfNotPresent(
                from: trainImagesURL, to: directory,
                filename: "train2017")
        }
        return loadJSON(
            directory,
            annotations: "annotations-train2017/instances_train2017.json",
            images: downloadImages ? "train2017" : nil)
    }

    public static func loadVal(
        to directory: URL = defaultDirectory(),
        downloadImages: Bool = false
    ) -> COCO {
        downloadIfNotPresent(
            from: valAnnotationsURL, to: directory,
            filename: "annotations-val2017")
        if downloadImages {
            downloadIfNotPresent(
                from: valImagesURL, to: directory,
                filename: "val2017")
        }
        return loadJSON(
            directory,
            annotations: "annotations-val2017/instances_val2017.json",
            images: downloadImages ? "val2017" : nil)
    }

    public static func loadTest(
        to directory: URL = defaultDirectory(),
        downloadImages: Bool = false
    ) -> COCO {
        downloadIfNotPresent(
            from: testAnnotationsURL, to: directory,
            filename: "annotations-test2017")
        if downloadImages {
            downloadIfNotPresent(
                from: testImagesURL, to: directory,
                filename: "test2017")
        }
        return loadJSON(
            directory,
            annotations: "annotations-test2017/image_info_test2017.json",
            images: downloadImages ? "test2017" : nil)
    }

    public static func loadTestDev(
        to directory: URL = defaultDirectory(),
        downloadImages: Bool = false
    ) -> COCO {
        downloadIfNotPresent(
            from: testDevAnnotationsURL, to: directory,
            filename: "annotations-test-dev2017")
        if downloadImages {
            downloadIfNotPresent(
                from: testImagesURL, to: directory,
                filename: "test2017")
        }
        return loadJSON(
            directory,
            annotations: "annotations-test-dev2017/image_info_test-dev2017.json",
            images: downloadImages ? "test2017" : nil)
    }
}
