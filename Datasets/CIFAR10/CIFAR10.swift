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
// "The CIFAR-10 dataset"
// Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.
// https://www.cs.toronto.edu/~kriz/cifar.html

import Foundation
import ModelSupport
import TensorFlow
import Batcher

public struct CIFAR10<Entropy: RandomNumberGenerator> {
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
    self.init(
      batchSize: batchSize,
      entropy: entropy,
      device: Device.default,
      remoteBinaryArchiveLocation: URL(
        string: "https://storage.googleapis.com/s4tf-hosted-binaries/datasets/CIFAR10/cifar-10-binary.tar.gz")!, 
      normalizing: true)
  }
  
  /// Creates an instance with `batchSize` on `device` using `remoteBinaryArchiveLocation`.
  ///
  /// - Parameters:
  ///   - entropy: a source of randomness used to shuffle sample ordering.  It  
  ///     will be stored in `self`, so if it is only pseudorandom and has value 
  ///     semantics, the sequence of epochs is deterministic and not dependent 
  ///     on other operations.
  ///   - normalizing: normalizes the batches with the mean and standard deviation
  ///     of the dataset iff `true`. Default value is `true`.
  public init(
    batchSize: Int,
    entropy: Entropy,
    device: Device,
    remoteBinaryArchiveLocation: URL, 
    localStorageDirectory: URL = DatasetUtilities.defaultDirectory
      .appendingPathComponent("CIFAR10", isDirectory: true), 
    normalizing: Bool
  ){
    downloadCIFAR10IfNotPresent(from: remoteBinaryArchiveLocation, to: localStorageDirectory)
    
    // Training data
    let trainingSamples = loadCIFARTrainingFiles(in: localStorageDirectory)
    training = TrainingEpochs(samples: trainingSamples, batchSize: batchSize, entropy: entropy)
      .lazy.map { (batches: Batches) -> LazyMapSequence<Batches, LabeledImage> in
        return batches.lazy.map{ makeBatch(samples: $0, normalizing: normalizing, device:device) }
      }
      
    // Validation data
    let validationSamples = loadCIFARTestFile(in: localStorageDirectory)
    validation = validationSamples.inBatches(of: batchSize).lazy.map {
      makeBatch(samples: $0, normalizing: normalizing, device:device)
    }
  }
}

extension CIFAR10: ImageClassificationData where Entropy == SystemRandomNumberGenerator {
  /// Creates an instance with `batchSize`.
  public init(batchSize: Int) {
    self.init(batchSize: batchSize, entropy: SystemRandomNumberGenerator())
  }
}

func downloadCIFAR10IfNotPresent(from location: URL, to directory: URL) {
  let downloadPath = directory.appendingPathComponent("cifar-10-batches-bin").path
  let directoryExists = FileManager.default.fileExists(atPath: downloadPath)
  let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: downloadPath)
  let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)

  guard !directoryExists || directoryEmpty else { return }

  let _ = DatasetUtilities.downloadResource(
    filename: "cifar-10-binary", fileExtension: "tar.gz",
    remoteRoot: location.deletingLastPathComponent(), localStorageDirectory: directory)
}

func loadCIFARFile(named name: String, in directory: URL) -> [(data: [UInt8], label: Int32)] {
  let path = directory.appendingPathComponent("cifar-10-batches-bin/\(name)").path

  let imageCount = 10000
  guard let fileContents = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
    printError("Could not read dataset file: \(name)")
    exit(-1)
  }
  guard fileContents.count == 30_730_000 else {
    printError(
      "Dataset file \(name) should have 30730000 bytes, instead had \(fileContents.count)")
    exit(-1)
  }

  var labeledImages: [(data: [UInt8], label: Int32)] = []

  let imageByteSize = 3073
  for imageIndex in 0..<imageCount {
    let baseAddress = imageIndex * imageByteSize
    let label = Int32(fileContents[baseAddress])
    let data = [UInt8](fileContents[(baseAddress + 1)..<(baseAddress + 3073)])
    labeledImages.append((data: data, label: label))
  }

  return labeledImages
}

func loadCIFARTrainingFiles(in localStorageDirectory: URL) -> [(data: [UInt8], label: Int32)] {
  let data = (1..<6).map {
    loadCIFARFile(named: "data_batch_\($0).bin", in: localStorageDirectory)
  }
  return data.reduce([], +)
}

func loadCIFARTestFile(in localStorageDirectory: URL) -> [(data: [UInt8], label: Int32)] {
  return loadCIFARFile(named: "test_batch.bin", in: localStorageDirectory)
}

fileprivate func makeBatch<BatchSamples: Collection>(
  samples: BatchSamples, normalizing: Bool, device:Device
) -> LabeledImage where BatchSamples.Element == (data: [UInt8], label: Int32) {
  let bytes = samples.lazy.map(\.data).reduce(into: [], +=)
  let images = Tensor<UInt8>(shape: [samples.count, 3, 32, 32], scalars: bytes, on:device)
  
  var imageTensor = Tensor<Float>(images.transposed(permutation: [0, 2, 3, 1]))
  imageTensor /= 255.0
  if normalizing {
    let mean = Tensor<Float>([0.4913996898, 0.4821584196, 0.4465309242], on:device)
    let std = Tensor<Float>([0.2470322324, 0.2434851280, 0.2615878417], on:device)
    imageTensor = (imageTensor - mean) / std
  }
  
  let labels = Tensor<Int32>(samples.map(\.label))
  return LabeledImage(data: imageTensor, label: labels)
}