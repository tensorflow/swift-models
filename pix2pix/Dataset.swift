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
import Datasets
import TensorFlow

public enum Pix2PixDatasetVariant: String {
    case facades

    public var url: URL {
        switch self {
        case .facades:
            return URL(string: 
                "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/facades.zip")!
        }
    }
}

public struct Pix2PixDataset<Entropy: RandomNumberGenerator> {
    public typealias Samples = [(source: Tensor<Float>, target: Tensor<Float>)]
    public typealias Batches = Slices<Sampling<Samples, ArraySlice<Int>>>
    public typealias PairedImageBatch = (source: Tensor<Float>, target: Tensor<Float>)
    public typealias Training = LazyMapSequence<
        TrainingEpochs<Samples, Entropy>, 
        LazyMapSequence<Batches, PairedImageBatch>
      >
    public typealias Testing = LazyMapSequence<
        Slices<Samples>, 
        PairedImageBatch
    >

    public let trainSamples: Samples
    public let testSamples: Samples
    public let training: Training
    public let testing: Testing

    public init(
        from rootDirPath: String? = nil,
        variant: Pix2PixDatasetVariant? = nil, 
        trainBatchSize: Int = 1,
        testBatchSize: Int = 1,
        entropy: Entropy) throws {
        
        let rootDirPath = rootDirPath ?? Pix2PixDataset.downloadIfNotPresent(
            variant: variant ?? .facades,
            to: DatasetUtilities.defaultDirectory.appendingPathComponent("pix2pix", isDirectory: true))
        let rootDirURL = URL(fileURLWithPath: rootDirPath, isDirectory: true)
        
        trainSamples = Array(zip(
            try Pix2PixDataset.loadSortedSamples(
                  from: rootDirURL.appendingPathComponent("trainB"),
                  fileIndexRetriever: "_"
                ), 
            try Pix2PixDataset.loadSortedSamples(
                  from: rootDirURL.appendingPathComponent("trainA"),
                  fileIndexRetriever: "_"
                )
        ))
        
        testSamples = Array(zip(
            try Pix2PixDataset.loadSortedSamples(
                  from: rootDirURL.appendingPathComponent("testB"),
                  fileIndexRetriever: "."
                ), 
            try Pix2PixDataset.loadSortedSamples(
                  from: rootDirURL.appendingPathComponent("testA"),
                  fileIndexRetriever: "."
                )
        ))

        training = TrainingEpochs(
            samples: trainSamples, 
            batchSize: trainBatchSize, 
            entropy: entropy
        ).lazy.map { (batches: Batches) -> LazyMapSequence<Batches, PairedImageBatch> in
            batches.lazy.map {
                (
                    source: Tensor<Float>($0.map(\.source)),
                    target: Tensor<Float>($0.map(\.target))
                )
            }
        }

        testing = testSamples.inBatches(of: testBatchSize)
            .lazy.map {
                (
                    source: Tensor<Float>($0.map(\.source)),
                    target: Tensor<Float>($0.map(\.target))
                )
            }
    }

    private static func downloadIfNotPresent(
            variant: Pix2PixDatasetVariant,
            to directory: URL) -> String {
        let rootDirPath = directory.appendingPathComponent(variant.rawValue).path

        let directoryExists = FileManager.default.fileExists(atPath: rootDirPath)
        let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: rootDirPath)
        let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)
        guard !directoryExists || directoryEmpty else { return rootDirPath }

        let _ = DatasetUtilities.downloadResource(
            filename: variant.rawValue, 
            fileExtension: "zip",
            remoteRoot: variant.url.deletingLastPathComponent(), 
            localStorageDirectory: directory)
        print("\(rootDirPath) downloaded.")

        return rootDirPath
    }

    private static func loadSortedSamples(
        from directory: URL, 
        fileIndexRetriever: String
    ) throws -> [Tensor<Float>] {
        return try FileManager.default
            .contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles])
            .filter { $0.pathExtension == "jpg" }
            .sorted {
                Int($0.lastPathComponent.components(separatedBy: fileIndexRetriever)[0])! <
                Int($1.lastPathComponent.components(separatedBy: fileIndexRetriever)[0])!
            }
            .map {
                Image(contentsOf: $0).tensor / 127.5 - 1.0
            }
    }
}

extension Pix2PixDataset where Entropy == SystemRandomNumberGenerator {
    public init(
        from rootDirPath: String? = nil,
        variant: Pix2PixDatasetVariant? = nil, 
        trainBatchSize: Int = 1,
        testBatchSize: Int = 1
    ) throws {
        try self.init(
            from: rootDirPath,
            variant: variant,
            trainBatchSize: trainBatchSize,
            testBatchSize: testBatchSize,
            entropy: SystemRandomNumberGenerator()
        )
  }
}
