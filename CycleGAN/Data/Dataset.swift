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


public enum CycleGANDatasetVariant: String {
    case horse2zebra

    public var url: URL {
        switch self {
        case .horse2zebra:
            return URL(string: 
                "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip")!
        }
    }
}

public struct CycleGANDataset<Entropy: RandomNumberGenerator> {
    public typealias Samples = [(domainA: Tensor<Float>, domainB: Tensor<Float>)]
    public typealias Batches = Slices<Sampling<Samples, ArraySlice<Int>>>
    public typealias PairedImageBatch = (domainA: Tensor<Float>, domainB: Tensor<Float>)
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
        variant: CycleGANDatasetVariant? = nil, 
        trainBatchSize: Int = 1,
        testBatchSize: Int = 1,
        entropy: Entropy) throws {
        
        let rootDirPath = rootDirPath ?? CycleGANDataset.downloadIfNotPresent(
            variant: variant ?? .horse2zebra,
            to: DatasetUtilities.defaultDirectory.appendingPathComponent("CycleGAN", isDirectory: true))
        let rootDirURL = URL(fileURLWithPath: rootDirPath, isDirectory: true)
        
        trainSamples = Array(zip(
            try CycleGANDataset.loadSamples(from: rootDirURL.appendingPathComponent("trainA")), 
            try CycleGANDataset.loadSamples(from: rootDirURL.appendingPathComponent("trainB"))))
        
        testSamples = Array(zip(
            try CycleGANDataset.loadSamples(from: rootDirURL.appendingPathComponent("testA")), 
            try CycleGANDataset.loadSamples(from: rootDirURL.appendingPathComponent("testB"))))

        training = TrainingEpochs(
            samples: trainSamples, 
            batchSize: trainBatchSize, 
            entropy: entropy
        ).lazy.map { (batches: Batches) -> LazyMapSequence<Batches, PairedImageBatch> in
            batches.lazy.map {
                (
                    domainA: Tensor<Float>($0.map(\.domainA)),
                    domainB: Tensor<Float>($0.map(\.domainB))
                )
            }
        }

        testing = testSamples.inBatches(of: testBatchSize)
            .lazy.map {
                (
                    domainA: Tensor<Float>($0.map(\.domainA)),
                    domainB: Tensor<Float>($0.map(\.domainB))
                )
            }
    }

    private static func downloadIfNotPresent(
            variant: CycleGANDatasetVariant,
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

        return rootDirPath
    }

    private static func loadSamples(from directory: URL) throws -> [Tensor<Float>] {
        return try FileManager.default
            .contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles])
            .filter { $0.pathExtension == "jpg" }
            .map {
                Image(jpeg: $0).tensor / 127.5 - 1.0
            }
    }
}

extension CycleGANDataset where Entropy == SystemRandomNumberGenerator {
    public init(
        from rootDirPath: String? = nil,
        variant: CycleGANDatasetVariant? = nil, 
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

