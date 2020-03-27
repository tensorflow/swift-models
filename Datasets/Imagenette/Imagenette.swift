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

// Original source:
// "Imagenette"
// Jeremy Howard
// https://github.com/fastai/imagenette

import Foundation
import ModelSupport
import TensorFlow
import Batcher

public typealias LazyDataSet = LazyMapSequence<[URL], TensorPair<Float, Int32>>

public struct Imagenette: ImageClassificationDataset {
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
                .appendingPathComponent("Imagenette", isDirectory: true)
    ) {
        do {
            training = Batcher<SourceDataSet>(
                on: try loadImagenetteTrainingDirectory(
                    inputSize: inputSize, outputSize: outputSize,
                    localStorageDirectory: localStorageDirectory),
                batchSize: batchSize, 
                shuffle: true)
            test = Batcher<SourceDataSet>(
                on: try loadImagenetteValidationDirectory(
                    inputSize: inputSize, outputSize: outputSize,
                    localStorageDirectory: localStorageDirectory),
                batchSize: batchSize)
        } catch {
            fatalError("Could not load Imagenette dataset: \(error)")
        }
    }
}

func downloadImagenetteIfNotPresent(to directory: URL, size: Imagenette.ImageSize) {
    let downloadPath = directory.appendingPathComponent("imagenette\(size.suffix)").path
    let directoryExists = FileManager.default.fileExists(atPath: downloadPath)
    let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: downloadPath)
    let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)

    guard !directoryExists || directoryEmpty else { return }

    let location = URL(
        string: "https://s3.amazonaws.com/fast-ai-imageclas/imagenette\(size.suffix).tgz")!
    let _ = DatasetUtilities.downloadResource(
        filename: "imagenette\(size.suffix)", fileExtension: "tgz",
        remoteRoot: location.deletingLastPathComponent(), localStorageDirectory: directory)
}

func exploreImagenetteDirectory(named name: String, in directory: URL, inputSize: Imagenette.ImageSize) throws -> [URL] {
    downloadImagenetteIfNotPresent(to: directory, size: inputSize)
    let path = directory.appendingPathComponent("imagenette\(inputSize.suffix)/\(name)")
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

func parentLabel(url: URL) -> String {
    return url.deletingLastPathComponent().lastPathComponent
}

func createLabelDict(urls: [URL]) -> [String: Int] {
    let allLabels = urls.map(parentLabel)
    let labels = Array(Set(allLabels)).sorted()
    return Dictionary(uniqueKeysWithValues: labels.enumerated().map{ ($0.element, $0.offset) })
}

func loadImagenetteDirectory(
    named name: String, in directory: URL, inputSize: Imagenette.ImageSize, outputSize: Int,
    labelDict: [String:Int]? = nil
) throws -> LazyDataSet {
    let urls = try exploreImagenetteDirectory(named: name, in: directory, inputSize: inputSize)
    let unwrappedLabelDict = labelDict ?? createLabelDict(urls: urls)
    return urls.lazy.map { (url: URL) -> TensorPair<Float, Int32> in
        TensorPair<Float, Int32>(
            first: Image(jpeg: url).resized(to: (outputSize, outputSize)).tensor[0] / 255.0,
            second: Tensor<Int32>(Int32(unwrappedLabelDict[parentLabel(url: url)]!))
        )    
    }
}

func loadImagenetteTrainingDirectory(
    inputSize: Imagenette.ImageSize, outputSize: Int, localStorageDirectory: URL, labelDict: [String:Int]? = nil
) throws
    -> LazyDataSet
{
    return try loadImagenetteDirectory(
        named: "train", in: localStorageDirectory, inputSize: inputSize, outputSize: outputSize, labelDict: labelDict)
}

func loadImagenetteValidationDirectory(
    inputSize: Imagenette.ImageSize, outputSize: Int, localStorageDirectory: URL, labelDict: [String:Int]? = nil
) throws
    -> LazyDataSet
{
    return try loadImagenetteDirectory(
        named: "val", in: localStorageDirectory, inputSize: inputSize, outputSize: outputSize, labelDict: labelDict)
}