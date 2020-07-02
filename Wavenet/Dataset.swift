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

import Datasets
import Foundation
import TensorFlow

public enum WavenetDatasetVariant: String {
    case vctk

    public var url: URL {
        switch self {
        case .vctk:
            return URL(string:
                "https://datashare.is.ed.ac.uk/bitstream/handle/10283/2651/VCTK-Corpus.zip")!
        }
    }
}

public struct WavenetDataset<Entropy: RandomNumberGenerator> {
    /// Type of the collection of non-collated batches.
    public typealias Batches = Slices<Sampling<[URL], ArraySlice<Int>>>
    /// The type of the training data, represented as a sequence of epochs, which
    /// are collection of batches.
    public typealias Training = LazyMapSequence<
        TrainingEpochs<[URL], Entropy>,
        LazyMapSequence<Batches, Tensor<Float>>
    >

    let training: Training
    let receptiveField: Int = 1
    let sampleSize: Int = 100_000
    let audioSampleRate: Int
    let audioReader: AudioReader

    public init(
        from rootDirPath: String? = nil,
        variant: WavenetDatasetVariant? = nil,
        audioSampleRate: Int,
        receptiveField: Int,
        batchSize _: Int,
        sampleSize: Int,
        entropy: Entropy
    ) throws {
        // Each audio file is sliced into fixed sized pieces.
        // Since each audio file is of a different size, each file
        // results in a different number of samples. Currently a single
        // file corresponds to a single batch since we don't have a good
        // way of reading part of a file. This will be fixed by using a
        // PaddingFIFOQueue. This needs to be added to
        // tensorflow/swift-apis before we can use it here
        print("Warning: Specified batch size is not honored at the moment.")
        let rootDirPath = rootDirPath ?? WavenetDataset.downloadIfNotPresent(
            variant: variant ?? .vctk,
            to: DatasetUtilities.defaultDirectory.appendingPathComponent("VCTK-Corpus", isDirectory: true)
        )
        let rootDirURL = URL(fileURLWithPath: rootDirPath, isDirectory: true).appendingPathComponent("VCTK-Corpus/wav48")

        audioReader = AudioReader(
            rootDir: rootDirURL,
            audioSampleRate: audioSampleRate,
            sampleSize: sampleSize
        )

        training = TrainingEpochs(
            samples: try! audioReader.loadDataFileNames(),
            batchSize: 1,
            entropy: entropy
        ).lazy.map { (batches: Batches) -> LazyMapSequence<Batches, Tensor<Float>> in
            batches.lazy.map {
                makeBatch(samples: $0, audioSampleRate: audioSampleRate,
                          receptiveField: receptiveField,
                          sampleSize: sampleSize).collated
            }
        }

        self.audioSampleRate = audioSampleRate
    }

    private static func downloadIfNotPresent(
        variant: WavenetDatasetVariant,
        to directory: URL
    ) -> String {
        let rootDirPath = directory.appendingPathComponent(variant.rawValue).path

        let directoryExists = FileManager.default.fileExists(atPath: rootDirPath)
        let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: rootDirPath)
        let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)
        guard !directoryExists || directoryEmpty else { return rootDirPath }

        _ = DatasetUtilities.downloadResource(
            filename: variant.rawValue,
            fileExtension: "zip",
            remoteRoot: variant.url.deletingLastPathComponent(),
            localStorageDirectory: directory
        )
        print("\(rootDirPath) downloaded.")

        return rootDirPath
    }
}

private func makeBatch<BatchSamples: Collection>(
    samples: BatchSamples, audioSampleRate: Int, receptiveField: Int,
    sampleSize: Int
) -> [Tensor<Float>] where BatchSamples.Element == URL {
    samples.reduce([]) {
        $0 + AudioReader.loadSamplesFromFile(from: $1, receptiveField: receptiveField, audioSampleRate: audioSampleRate,
                                             sampleSize: sampleSize)
    }
}
