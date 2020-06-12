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

public struct Imagewoof<Entropy: RandomNumberGenerator> {
  /// Type of the collection of non-collated batches.
  public typealias Batches = Slices<Sampling<[(file: URL, label: Int32)], ArraySlice<Int>>>
  /// The type of the training data, represented as a sequence of epochs, which
  /// are collection of batches.
  public typealias Training = LazyMapSequence<
    TrainingEpochs<[(file: URL, label: Int32)], Entropy>,
    LazyMapSequence<Batches, LabeledImage>
  >
  /// The type of the validation data, represented as a collection of batches.
  public typealias Validation = LazyMapSequence<Slices<[(file: URL, label: Int32)]>, LabeledImage>
  /// The training epochs.
  public let training: Training
  /// The validation batches.
  public let validation: Validation

  /// Creates an instance with `batchSize`.
  ///
  /// - Parameters:
  ///   - batchSize: Number of images provided per batch.
  ///   - entropy: A source of randomness used to shuffle sample
  ///     ordering.  It  will be stored in `self`, so if it is only pseudorandom
  ///     and has value semantics, the sequence of epochs is deterministic and not
  ///     dependent on other operations.
  ///   - device: The Device on which resulting Tensors from this dataset will be placed, as well
  ///     as where the latter stages of any conversion calculations will be performed.
  public init(batchSize: Int, entropy: Entropy, device: Device) {
    self.init(
      batchSize: batchSize, entropy: entropy, device: device, inputSize: ImagenetteSize.resized320,
      outputSize: 224)
  }

  /// Creates an instance with `batchSize` on `device` using `remoteBinaryArchiveLocation`.
  ///
  /// - Parameters:
  ///   - batchSize: Number of images provided per batch.
  ///   - entropy: A source of randomness used to shuffle sample ordering.  It
  ///     will be stored in `self`, so if it is only pseudorandom and has value
  ///     semantics, the sequence of epochs is deterministic and not dependent
  ///     on other operations.
  ///   - device: The Device on which resulting Tensors from this dataset will be placed, as well
  ///     as where the latter stages of any conversion calculations will be performed.
  ///   - inputSize: Which Imagenette image size variant to use.
  ///   - outputSize: The square width and height of the images returned from this dataset.
  ///   - localStorageDirectory: Where to place the downloaded and unarchived dataset.
  public init(
    batchSize: Int, entropy: Entropy, device: Device, inputSize: ImagenetteSize,
    outputSize: Int,
    localStorageDirectory: URL = DatasetUtilities.defaultDirectory
      .appendingPathComponent("Imagewoof", isDirectory: true)
  ) {
    do {
      // Training data
      let trainingSamples = try loadImagenetteTrainingDirectory(
        inputSize: inputSize, localStorageDirectory: localStorageDirectory, base: "imagewoof")

      let mean = Tensor<Float>([0.485, 0.456, 0.406], on: device)
      let standardDeviation = Tensor<Float>([0.229, 0.224, 0.225], on: device)

      training = TrainingEpochs(samples: trainingSamples, batchSize: batchSize, entropy: entropy)
        .lazy.map { (batches: Batches) -> LazyMapSequence<Batches, LabeledImage> in
          return batches.lazy.map {
            makeImagenetteBatch(
              samples: $0, outputSize: outputSize, mean: mean, standardDeviation: standardDeviation,
              device: device)
          }
        }

      // Validation data
      let validationSamples = try loadImagenetteValidationDirectory(
        inputSize: inputSize, localStorageDirectory: localStorageDirectory, base: "imagewoof")

      validation = validationSamples.inBatches(of: batchSize).lazy.map {
        makeImagenetteBatch(
          samples: $0, outputSize: outputSize, mean: mean, standardDeviation: standardDeviation,
          device: device)
      }
    } catch {
      fatalError("Could not load Imagewoof dataset: \(error)")
    }
  }
}

extension Imagewoof: ImageClassificationData where Entropy == SystemRandomNumberGenerator {
  /// Creates an instance with `batchSize`.
  public init(batchSize: Int, on device: Device = Device.default) {
    self.init(batchSize: batchSize, entropy: SystemRandomNumberGenerator(), device: device)
  }

  public init(
    batchSize: Int, inputSize: ImagenetteSize, outputSize: Int, on device: Device = Device.default
  ) {
    self.init(
      batchSize: batchSize, entropy: SystemRandomNumberGenerator(), device: device,
      inputSize: inputSize, outputSize: outputSize)
  }
}
