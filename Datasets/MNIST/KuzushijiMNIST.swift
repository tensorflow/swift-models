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
// "KMNIST Dataset" (created by CODH), https://arxiv.org/abs/1812.01718
// adapted from "Kuzushiji Dataset" (created by NIJL and others), doi:10.20676/00000341

import Foundation
import TensorFlow
import Batcher

public struct KuzushijiMNIST<Entropy: RandomNumberGenerator> {
  /// Type of the collection of non-collated batches.
  public typealias Batches = Slices<Sampling<[(data: [UInt8], label: Int32)], ArraySlice<Int>>>
  /// The type of the training data, represented as a sequence of epochs, which
  /// are collection of batches.
  public typealias Training = LazyMapSequence<
    TrainingEpochs<[(data: [UInt8], label: Int32)], Entropy>,
    LazyMapSequence<Batches, LabeledImage>
  >
  /// The type of the validation data, represented as a collection of batches.
  public typealias Validation = LazyMapSequence<Slices<[(data: [UInt8], label: Int32)]>, LabeledImage>
  /// The training epochs.
  public let training: Training
  /// The validation batches.
  public let validation: Validation

  /// Creates an instance with `batchSize`.
  ///
  /// - Parameter entropy: a source of randomness used to shuffle sample 
  ///   ordering.  It  will be stored in `self`, so if it is only pseudorandom 
  ///   and has value semantics, the sequence of epochs is deterministic and not 
  ///   dependent on other operations.
  public init(batchSize: Int, entropy: Entropy) {
    self.init(batchSize: batchSize, device: Device.default, entropy: entropy,
              flattening: false, normalizing: false)
  }

  /// Creates an instance with `batchSize` on `device`.
  ///
  /// - Parameters:
  ///   - entropy: a source of randomness used to shuffle sample ordering.  It  
  ///     will be stored in `self`, so if it is only pseudorandom and has value 
  ///     semantics, the sequence of epochs is deterministic and not dependent 
  ///     on other operations.
  ///   - flattening: flattens the data to be a 2d-tensor iff `true. The default value
  ///     is `false`.
  ///   - normalizing: normalizes the batches to have values from -1.0 to 1.0 iff `true`.
  ///     The default value is `false`.
  ///   - localStorageDirectory: the directory in which the dataset is stored.
  public init(
    batchSize: Int, device: Device, entropy: Entropy, flattening: Bool = false, 
    normalizing: Bool = false, 
    localStorageDirectory: URL = DatasetUtilities.defaultDirectory
      .appendingPathComponent("KuzushijiMNIST", isDirectory: true)
  ) {
    training = TrainingEpochs(
      samples: fetchMNISTDataset(
        localStorageDirectory: localStorageDirectory,
        remoteBaseDirectory: "https://storage.googleapis.com/s4tf-hosted-binaries/datasets/KMNIST",
        imagesFilename: "train-images-idx3-ubyte",
        labelsFilename: "train-labels-idx1-ubyte"),
      batchSize: batchSize, entropy: entropy
    ).lazy.map { (batches: Batches) -> LazyMapSequence<Batches, LabeledImage> in
      return batches.lazy.map{ makeMNISTBatch(
        samples: $0, flattening: flattening, normalizing: normalizing, device: device
      )}
    }
    
    validation = fetchMNISTDataset(
      localStorageDirectory: localStorageDirectory,
      remoteBaseDirectory: "https://storage.googleapis.com/s4tf-hosted-binaries/datasets/KMNIST",
      imagesFilename: "t10k-images-idx3-ubyte",
      labelsFilename: "t10k-labels-idx1-ubyte"
    ).inBatches(of: batchSize).lazy.map {
      makeMNISTBatch(samples: $0, flattening: flattening, normalizing: normalizing, 
                     device: device)
    }
  }
}

extension KuzushijiMNIST: ImageClassificationData where Entropy == SystemRandomNumberGenerator {
  /// Creates an instance with `batchSize`.
  public init(batchSize: Int) {
    self.init(batchSize: batchSize, entropy: SystemRandomNumberGenerator())
  }
}