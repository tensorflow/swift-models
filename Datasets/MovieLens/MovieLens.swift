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
// http://files.grouplens.org/datasets/movielens/
// F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
// History and Context. ACM Transactions on Interactive Intelligent
// Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages.
// DOI=http://dx.doi.org/10.1145/2827872

import Foundation
import TensorFlow

extension Sequence where Element: Collection {
    subscript(column column: Element.Index) -> [Element.Iterator.Element] {
        return map { $0[column] }
    }
}
extension Sequence where Iterator.Element: Hashable {
    func unique() -> [Iterator.Element] {
        var seen: Set<Iterator.Element> = []
        return filter { seen.insert($0).inserted }
    }
}

public struct MovieLens<Entropy: RandomNumberGenerator> {
    public let trainUsers: [Float]
    public let testUsers: [Float]
    public let testData: [[Float]]
    public let items: [Float]
    public let numUsers: Int
    public let numItems: Int  
    public let user2id: [Float: Int]
    public let id2user: [Int: Float]
    public let item2id: [Float: Int]
    public let id2item: [Int: Float]
    public let trainNegSampling: Tensor<Float>

    public typealias Samples = [TensorPair<Int32, Float>]
    public typealias Batches = Slices<Sampling<Samples, ArraySlice<Int>>>
    public typealias BatchedTensorPair = TensorPair<Int32, Float>
    public typealias Training = LazyMapSequence<
        TrainingEpochs<Samples, Entropy>, 
        LazyMapSequence<Batches, BatchedTensorPair>
      >
    public let trainMatrix: Samples
    public let training: Training

    static func downloadMovieLensDatasetIfNotPresent() -> URL {
        let localURL = DatasetUtilities.defaultDirectory.appendingPathComponent(
            "MovieLens", isDirectory: true)
        let dataFolder = DatasetUtilities.downloadResource(
            filename: "ml-100k",
            fileExtension: "zip",
            remoteRoot: URL(string: "http://files.grouplens.org/datasets/movielens/")!,
            localStorageDirectory: localURL)

        return dataFolder
    }

    public init(
            trainBatchSize: Int = 1024, 
            entropy: Entropy) {
        let trainFiles = try! String(
            contentsOf: MovieLens.downloadMovieLensDatasetIfNotPresent().appendingPathComponent(
                "u1.base"), encoding: .utf8)
        let testFiles = try! String(
            contentsOf: MovieLens.downloadMovieLensDatasetIfNotPresent().appendingPathComponent(
                "u1.test"), encoding: .utf8)

        let trainData: [[Float]] = trainFiles.split(separator: "\n").map {
            String($0).split(separator: "\t").compactMap { Float(String($0)) }
        }
        let testData: [[Float]] = testFiles.split(separator: "\n").map {
            String($0).split(separator: "\t").compactMap { Float(String($0)) }
        }

        let trainUsers = trainData[column: 0].unique()
        let testUsers = testData[column: 0].unique()

        let items = trainData[column: 1].unique()

        let userIndex = 0...trainUsers.count - 1
        let user2id = Dictionary(uniqueKeysWithValues: zip(trainUsers, userIndex))
        let id2user = Dictionary(uniqueKeysWithValues: zip(userIndex, trainUsers))

        let itemIndex = 0...items.count - 1
        let item2id = Dictionary(uniqueKeysWithValues: zip(items, itemIndex))
        let id2item = Dictionary(uniqueKeysWithValues: zip(itemIndex, items))

        var trainNegSampling = Tensor<Float>(zeros: [trainUsers.count, items.count])

        var dataset: [TensorPair<Int32, Float>] = []

        for element in trainData {
            let uIndex = user2id[element[0]]!
            let iIndex = item2id[element[1]]!
            let rating = element[2]
            if rating > 0 {
                trainNegSampling[uIndex][iIndex] = Tensor(1.0)
            }
        }

        for element in trainData {
            let uIndex = user2id[element[0]]!
            let iIndex = item2id[element[1]]!
            let x = Tensor<Int32>([Int32(uIndex), Int32(iIndex)])
            dataset.append(TensorPair<Int32, Float>(first: x, second: [1]))

            for _ in 0...3 {
                var iIndex = Int.random(in: itemIndex)
                while trainNegSampling[uIndex][iIndex].scalarized() == 1.0 {
                    iIndex = Int.random(in: itemIndex)
                }
                let x = Tensor<Int32>([Int32(uIndex), Int32(iIndex)])
                dataset.append(TensorPair<Int32, Float>(first: x, second: [0]))
            }
        }

        self.testData = testData
        self.numUsers = trainUsers.count
        self.numItems = items.count
        self.trainUsers = trainUsers
        self.testUsers = testUsers
        self.items = items
        self.user2id = user2id
        self.id2user = id2user
        self.item2id = item2id
        self.id2item = id2item
        self.trainNegSampling = trainNegSampling

        self.trainMatrix = dataset
        self.training = TrainingEpochs(
            samples: trainMatrix, 
            batchSize: trainBatchSize, 
            entropy: entropy
        ).lazy.map { (batches: Batches) -> LazyMapSequence<Batches, BatchedTensorPair> in
            batches.lazy.map {
                TensorPair<Int32, Float> (
                    first: Tensor<Int32>($0.map(\.first)),
                    second: Tensor<Float>($0.map(\.second))
                )
            }
        }
    }
}

extension MovieLens where Entropy == SystemRandomNumberGenerator {
    public init(trainBatchSize: Int = 1024) {
        self.init(
            trainBatchSize: trainBatchSize, 
            entropy: SystemRandomNumberGenerator())
    }
}
