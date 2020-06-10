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

import Datasets
import TensorFlow

public class SyntheticImageDataset<Entropy: RandomNumberGenerator> {
  /// Type of the collection of non-collated batches.
  public typealias Batches = Slices<Sampling<Range<Int>, ArraySlice<Int>>>
  /// The type of the training data, represented as a sequence of epochs, which
  /// are collection of batches.
  public typealias Training = LazyMapSequence<
    TrainingEpochs<Range<Int>, Entropy>,
    LazyMapSequence<Batches, LabeledImage>
  >
  /// The type of the validation data, represented as a collection of batches.
  public typealias Validation = LazyMapSequence<Slices<Range<Int>>, LabeledImage>
  /// The training epochs.
  public let training: Training
  /// The validation batches.
  public let validation: Validation

  /// Creates an instance with `batchSize` on `device` using `remoteBinaryArchiveLocation`.
  ///
  /// - Parameters:
  ///   - entropy: a source of randomness used to shuffle sample ordering.  It
  ///     will be stored in `self`, so if it is only pseudorandom and has value
  ///     semantics, the sequence of epochs is deterministic and not dependent
  ///     on other operations.
  ///   - labels: the number of output labels in the classification dataset.
  ///   - dimensions: the height x width x depth dimensions of the generated images.

  public init(
    batchSize: Int,
    labels: Int,
    dimensions: [Int],
    entropy: Entropy,
    device: Device
  ) {
    precondition(labels > 0)
    precondition(dimensions.count == 3)

    // Training data
    training = TrainingEpochs(samples: (0..<batchSize), batchSize: batchSize, entropy: entropy)
      .lazy.map { (batches: Batches) -> LazyMapSequence<Batches, LabeledImage> in
        return batches.lazy.map {
          makeSyntheticBatch(samples: $0, dimensions: dimensions, labels: labels, device: device)
        }
      }

    // Validation data
    validation = (0..<batchSize).inBatches(of: batchSize).lazy.map {
      makeSyntheticBatch(samples: $0, dimensions: dimensions, labels: labels, device: device)
    }
  }
}

fileprivate func makeSyntheticBatch<BatchSamples: Collection>(
  samples: BatchSamples, dimensions: [Int], labels: Int, device: Device
) -> LabeledImage where BatchSamples.Element == Int {
  let syntheticImageBatch = Tensor<Float>(
    glorotUniform: TensorShape([samples.count] + dimensions), on: device)

  let syntheticLabels = Tensor<Int32>(
    samples.map { _ -> Int32 in
      Int32.random(in: 0..<Int32(labels))
    }, on: device)

  return LabeledImage(data: syntheticImageBatch, label: syntheticLabels)
}
