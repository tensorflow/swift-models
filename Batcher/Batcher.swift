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

import TensorFlow

// Return the input with no operation done
public func identity<C>(_ x: C) -> C { return x }

// Default sample on a collection
// Returns indices from 0 to dataset.count, potentially shuffled
public func defaultSample<C: Collection>(on dataset: inout C, shuffled: Bool) -> [Int] {
    return shuffled ? Array(0..<dataset.count).shuffled() : Array(0..<dataset.count)
}

// Default collate function for samples that conform to Collatable
public func defaultCollate<S: Collatable>(_ batch: [S]) -> S {
    return S(collating: batch)
}

// Main struct to collate the samples from a dataset into a batch
public struct Batcher<C: Collection> where C.Index == Int {
    // Dataset to get the batches from
    public var dataset: C
    // Size of the the batch
    public var batchSize: Int
    // Number of workers to use in parallel to fetch the samples
    public var numWorkers: Int = 1
    // Shuffle the dataset at each iteration
    public var shuffle: Bool = false
    // Drop the last batch if it has less elements than batchSize
    public var dropLast: Bool = false
    // Hook to customize the way indices are sampled at each iteration
    public let sampleIndices: (inout C, Bool) -> [Int]
    // Hook to add padding to the samples before they are collated
    public let padSamples: ([C.Element]) -> [C.Element]
    // Hook to customize how the samples are collated
    public let collateSamples: ([C.Element]) -> C.Element
    
    // Length of the batcher (number of batches it contains)
    public var count: Int {
        let nSamples = dataset.count
        return nSamples / batchSize + (nSamples % batchSize == 0 || dropLast ? 0 : 1)
    }
    
    public init(
        on dataset: C, 
        batchSize: Int, 
        numWorkers: Int = 1, 
        shuffle: Bool = false, 
        dropLast: Bool = false,
        sampleIndices: @escaping (inout C, Bool) -> [Int] = defaultSample,
        padSamples: @escaping ([C.Element]) -> [C.Element] = identity,
        collateSamples: @escaping ([C.Element]) -> C.Element
    ) {
        self.dataset = dataset
        self.batchSize = batchSize
        self.numWorkers = numWorkers
        self.shuffle = shuffle
        self.dropLast = dropLast
        self.sampleIndices = sampleIndices
        self.padSamples = padSamples
        self.collateSamples = collateSamples
    }
    
    // To iterate through the batches
    public func sequenced() -> BatchIterator<C> {
        return BatchIterator(self)
    }
}

// Iterator through a Batcher
public struct BatchIterator<C: Collection>: IteratorProtocol, Sequence where C.Index == Int{
    // Batcher to iterate through
    var b: Batcher<C>
    // Indices that will be used to go through the dataset of b
    let indices: [Int]
    // The length of the underlying dataset
    let samplesCount: Int
    // Where we are at in the dataset
    var pos: Int = 0
    
    init(_ b: Batcher<C>) { 
        self.b = b
        indices = b.sampleIndices(&self.b.dataset, b.shuffle)
        samplesCount = b.dataset.count
        pos = 0
    }
    
    // Returns the next batch
    public mutating func next() -> C.Element? {
        guard pos < samplesCount else { return nil }
        let end = Swift.min(pos + b.batchSize, samplesCount)
        if (end - pos) < b.batchSize && b.dropLast { return nil }
        // The idea is to have samples processed and collated on the CPU before moving to the host.
        // This part has not been optimized yet
        return withDevice(.cpu) { () -> C.Element in
            let samples = Array(pos..<end).concurrentMap(nthreads: b.numWorkers) {
                b.dataset[indices[$0]]
            }
            pos = end
            return b.collateSamples(b.padSamples(samples))
        }
    }
}

// Add default collateSamples when the dataset elements conform to Collatable
public extension Batcher where C.Element: Collatable {
    init(
        on dataset: C, 
        batchSize: Int, 
        numWorkers: Int = 1, 
        shuffle: Bool = false, 
        dropLast: Bool = false,
        sampleIndices: @escaping (inout C, Bool) -> [Int] = defaultSample,
        padSamples: @escaping ([C.Element]) -> [C.Element] = identity,
        collateSamples: @escaping ([C.Element]) -> C.Element = defaultCollate
    ) {
        self.dataset = dataset
        self.batchSize = batchSize
        self.numWorkers = numWorkers
        self.shuffle = shuffle
        self.dropLast = dropLast
        self.sampleIndices = sampleIndices
        self.padSamples = padSamples
        self.collateSamples = collateSamples
    }
}