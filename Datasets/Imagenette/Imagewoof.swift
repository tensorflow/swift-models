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

// Original source:
// "Imagenette"
// Jeremy Howard
// https://github.com/fastai/Imagenette

import Foundation
import ModelSupport
import TensorFlow
import Batcher

public struct Imagewoof: ImageClassificationDataset {
    public typealias SourceDataSet = LazyDataSet
    public let training: Batcher<SourceDataSet>
    public let test: Batcher<SourceDataSet>

    public enum ImageSize {
        case full
        case resized160
        case resized320

        var suffix: String {
            switch self {
            case .full: return ""
            case .resized160: return "-160"
            case .resized320: return "-320"
            }
        }
    }

    public init(batchSize: Int) {
        self.init(batchSize: batchSize, inputSize: .resized320, outputSize: 224)
    }

    public init(
        batchSize: Int,
        inputSize: ImageSize, outputSize: Int,
        localStorageDirectory: URL = DatasetUtilities.defaultDirectory
            .appendingPathComponent("Imagewoof", isDirectory: true)
    ) {
        do {
            training = Batcher<SourceDataSet>(
                on: try loadImagewoofTrainingDirectory(
                    inputSize: inputSize, outputSize: outputSize,
                    localStorageDirectory: localStorageDirectory),
                batchSize: batchSize, 
                shuffle: true)
            test = Batcher<SourceDataSet>(
                on: try loadImagewoofValidationDirectory(
                    inputSize: inputSize, outputSize: outputSize,
                    localStorageDirectory: localStorageDirectory),
                batchSize: batchSize)
        } catch {
            fatalError("Could not load Imagenette dataset: \(error)")
        }
    }
}

func downloadImagewoofIfNotPresent(to directory: URL, size: Imagewoof.ImageSize) {
    let downloadPath = directory.appendingPathComponent("imagewoof\(size.suffix)").path
    let directoryExists = FileManager.default.fileExists(atPath: downloadPath)
    let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: downloadPath)
    let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)

    guard !directoryExists || directoryEmpty else { return }

    let location = URL(
        string: "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof\(size.suffix).tgz")!
    let _ = DatasetUtilities.downloadResource(
        filename: "imagewoof\(size.suffix)", fileExtension: "tgz",
        remoteRoot: location.deletingLastPathComponent(), localStorageDirectory: directory)
}

func exploreImagewoofDirectory(named name: String, in directory: URL, inputSize: Imagewoof.ImageSize) throws -> [URL] {
    downloadImagewoofIfNotPresent(to: directory, size: inputSize)
    let path = directory.appendingPathComponent("imagewoof\(inputSize.suffix)/\(name)")
    let dirContents = try FileManager.default.contentsOfDirectory(
        at: path, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles])

    var urls: [URL] = []
    for directoryURL in dirContents {
        let subdirContents = try FileManager.default.contentsOfDirectory(
            at: directoryURL, includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles])
        urls += subdirContents
    }
    return urls
}

func loadImagewoofDirectory(
    named name: String, in directory: URL, inputSize: Imagewoof.ImageSize, outputSize: Int,
    labelDict: [String:Int]? = nil
) throws -> LazyDataSet {
    let urls = try exploreImagewoofDirectory(named: name, in: directory, inputSize: inputSize)
    let unwrappedLabelDict = labelDict ?? createLabelDict(urls: urls)
    return urls.lazy.map { (url: URL) -> TensorPair<Float, Int32> in
        TensorPair<Float, Int32>(
            first: Image(jpeg: url).resized(to: (outputSize, outputSize)).tensor / 255.0,
            second: Tensor<Int32>(Int32(unwrappedLabelDict[parentLabel(url: url)]!))
        )    
    }
}

func loadImagewoofTrainingDirectory(
    inputSize: Imagewoof.ImageSize, outputSize: Int, localStorageDirectory: URL, labelDict: [String:Int]? = nil
) throws
    -> LazyDataSet
{
    return try loadImagewoofDirectory(
        named: "train", in: localStorageDirectory, inputSize: inputSize, outputSize: outputSize, labelDict: labelDict)
}

func loadImagewoofValidationDirectory(
    inputSize: Imagewoof.ImageSize, outputSize: Int, localStorageDirectory: URL, labelDict: [String:Int]? = nil
) throws
    -> LazyDataSet
{
    return try loadImagewoofDirectory(
        named: "val", in: localStorageDirectory, inputSize: inputSize, outputSize: outputSize, labelDict: labelDict)
}