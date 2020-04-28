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

// Original Source
// "The Oxford-IIIT Pet Dataset"
// Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman and C. V. Jawahar
// https://www.robots.ox.ac.uk/~vgg/data/pets/

import Batcher
import Foundation
import ModelSupport
import PythonKit
import TensorFlow

let os = Python.import("os")

public struct OxfordIIITPets: ImageSegmentationDataset {
    public typealias SourceDataSet = LazyDataSet
    public let training: Batcher<SourceDataSet>
    public let test: Batcher<SourceDataSet>

    public init(batchSize: Int) {
        self.init(batchSize: batchSize, imageSize: 224)
    }

    public init(
        batchSize: Int,
        localStorageDirectory: URL = DatasetUtilities.defaultDirectory
            .appendingPathComponent("OxfordIIITPets", isDirectory: true),
        imageSize: Int
    ) {
        do {
            training = Batcher<SourceDataSet>(
                on: try loadOxfordIITPetsTraining(
                    imageSize: imageSize,
                    localStorageDirectory: localStorageDirectory
                ),
                batchSize: batchSize,
                shuffle: true)
            test = Batcher<SourceDataSet>(
                on: try loadOxfordIIITPetsValidation(
                    imageSize: imageSize,
                    localStorageDirectory: localStorageDirectory
                ),
                batchSize: batchSize)
        } catch {
            fatalError("Could not load Oxford IIIT Pets dataset: \(error)")
        }
    }
}

func downloadOxfordIIITPetsIfNotPresent(to directory: URL) {
    let downloadPath = directory.appendingPathComponent("images", isDirectory: true).path
    let directoryExists = FileManager.default.fileExists(atPath: downloadPath)
    let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: downloadPath)
    let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)

    guard !directoryExists || directoryEmpty else { return }

    let remoteRoot = URL(string: "https://www.robots.ox.ac.uk/~vgg/data/pets/data/")!

    os.system("wget \(remoteRoot.appendingPathComponent("images.tar.gz")) -P \(directory.path)")
    extractArchive(
        at: directory.appendingPathComponent("images.tar.gz"), to: directory,
        fileExtension: "tar.gz", deleteArchiveWhenDone: true)
    os.system(
        "wget \(remoteRoot.appendingPathComponent("annotations.tar.gz")) -P \(directory.path)")
    extractArchive(
        at: directory.appendingPathComponent("annotations.tar.gz"), to: directory,
        fileExtension: "tar.gz", deleteArchiveWhenDone: true)

    /// let _ = DatasetUtilities.downloadResource(
    ///     filename: "images", fileExtension: "tar.gz",
    ///     remoteRoot: remoteRoot, localStorageDirectory: directory
    /// )

    /// let _ = DatasetUtilities.downloadResource(
    ///     filename: "annotations", fileExtension: "tar.gz",
    ///     remoteRoot: remoteRoot, localStorageDirectory: directory
    /// )

}

func loadOxfordIIITPets(filename: String, in directory: URL, imageSize: Int) throws -> LazyDataSet {
    downloadOxfordIIITPetsIfNotPresent(to: directory)
    let imageURLs = getImageURLs(filename: filename, directory: directory)
    return imageURLs.lazy.map { (imageURL: URL) -> TensorPair<Float, Int32> in
        TensorPair<Float, Int32>(
            first:
                Image(jpeg: imageURL).resized(to: (imageSize, imageSize)).tensor[0..., 0..., 0..<3]
                / 255.0,
            second: Tensor<Int32>(
                Image(jpeg: makeAnnotationURL(imageURL: imageURL, directory: directory)).resized(
                    to: (imageSize, imageSize)
                ).tensor[0..., 0..., 0...0] - 1
            )
        )

    }
}

func makeAnnotationURL(imageURL: URL, directory: URL) -> URL {
    let filename = imageURL.deletingPathExtension().lastPathComponent
    return directory.appendingPathComponent("annotations/trimaps/\(filename).png")
}

func getImageURLs(filename: String, directory: URL) -> [URL] {
    let filePath = directory.appendingPathComponent("annotations/\(filename)")
    let imagesRootDirectory = directory.appendingPathComponent("images", isDirectory: true)
    let fileContents = try? String(contentsOf: filePath)
    let imageDetails = fileContents!.split(separator: "\n")
    return imageDetails.map {
        let imagename = String($0[..<$0.firstIndex(of: " ")!])
        return imagesRootDirectory.appendingPathComponent("\(imagename).jpg")
    }
}

func loadOxfordIITPetsTraining(
    imageSize: Int, localStorageDirectory: URL
) throws
    -> LazyDataSet
{
    return try loadOxfordIIITPets(
        filename: "trainval.txt", in: localStorageDirectory, imageSize: imageSize)
}

func loadOxfordIIITPetsValidation(
    imageSize: Int, localStorageDirectory: URL
) throws
    -> LazyDataSet
{
    return try loadOxfordIIITPets(
        filename: "test.txt", in: localStorageDirectory, imageSize: imageSize)
}
