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
// "The CIFAR-100 dataset"
// Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.
// https://www.cs.toronto.edu/~kriz/cifar.html

import Foundation
import ModelSupport
import TensorFlow

public struct CIFAR100<Entropy: RandomNumberGenerator> {
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
        string: "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz")!, 
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
      .appendingPathComponent("CIFAR100", isDirectory: true), 
    normalizing: Bool
  ){
    downloadCIFAR100IfNotPresent(from: remoteBinaryArchiveLocation, to: localStorageDirectory)
    
    // Training data
    let trainingSamples = loadCIFAR100TrainingFiles(in: localStorageDirectory)
    training = TrainingEpochs(samples: trainingSamples, batchSize: batchSize, entropy: entropy)
      .lazy.map { (batches: Batches) -> LazyMapSequence<Batches, LabeledImage> in
        return batches.lazy.map{ makeBatch(samples: $0, normalizing: normalizing, device: device) }
      }
      
    // Validation data
    let validationSamples = loadCIFAR100TestFiles(in: localStorageDirectory)
    validation = validationSamples.inBatches(of: batchSize).lazy.map {
      makeBatch(samples: $0, normalizing: normalizing, device: device)
    }
  }
}

extension CIFAR100: ImageClassificationData where Entropy == SystemRandomNumberGenerator {
  /// Creates an instance with `batchSize`.
  public init(batchSize: Int, on: Device) {
    self.init(batchSize: batchSize, entropy: SystemRandomNumberGenerator())
  }
}

func downloadCIFAR100IfNotPresent(from location: URL, to directory: URL) {
  let downloadPath = directory.path
  let directoryExists = FileManager.default.fileExists(atPath: downloadPath)
  let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: downloadPath)
  let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)

  guard !directoryExists || directoryEmpty else { return }

  let _ = DatasetUtilities.downloadResource(
    filename: "cifar-100-binary", fileExtension: "tar.gz",
    remoteRoot: location.deletingLastPathComponent(), localStorageDirectory: directory)
}

func loadCIFAR100File(named name: String, in directory: URL) -> [(data: [UInt8], label: Int32)] {
  let path = directory.appendingPathComponent("cifar-100-binary/\(name)").path


  var imageCount = 50000
  guard let fileContents = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
    printError("Could not read dataset file: \(name)")
    exit(-1)
  }
  if name.contains("test") {
      guard fileContents.count == 307_400_00 else {
      printError(
        "Dataset file \(name) should have 307_400_00 bytes, instead had \(fileContents.count)")
      exit(-1)
    }
    imageCount = 10000
  }
  else {
      guard fileContents.count == 153_700_000 else {
        printError(
          "Dataset file \(name) should have 15370000 bytes, instead had \(fileContents.count)")
        exit(-1)
      }
  }

  var labeledImages: [(data: [UInt8], label: Int32)] = []

  let imageByteSize = 3074
  for imageIndex in 0..<imageCount {
    let baseAddress = imageIndex * imageByteSize
    let label = Int32(fileContents[baseAddress + 1])
    let data = [UInt8](fileContents[(baseAddress + 2)..<(baseAddress + 3074)])
    labeledImages.append((data: data, label: label))
  }

  return labeledImages
}

func loadCIFAR100TrainingFiles(in localStorageDirectory: URL) -> [(data: [UInt8], label: Int32)] {
  return loadCIFAR100File(named: "train.bin", in: localStorageDirectory)
}

func loadCIFAR100TestFiles(in localStorageDirectory: URL) -> [(data: [UInt8], label: Int32)] {
  return loadCIFAR100File(named: "test.bin", in: localStorageDirectory)
}

fileprivate func makeBatch<BatchSamples: Collection>(
  samples: BatchSamples, normalizing: Bool, device: Device
) -> LabeledImage where BatchSamples.Element == (data: [UInt8], label: Int32) {
  let bytes = samples.lazy.map(\.data).reduce(into: [], +=)
  let images = Tensor<UInt8>(shape: [samples.count, 3, 32, 32], scalars: bytes, on: device)
  
  var imageTensor = Tensor<Float>(images.transposed(permutation: [0, 2, 3, 1]))
  imageTensor /= 255.0
  if normalizing {
    let mean = Tensor<Float>([0.5071, 0.4867, 0.4408], on: device)
    let std = Tensor<Float>([0.2675, 0.2565, 0.2761], on: device)
    imageTensor = (imageTensor - mean) / std
  }
  
  let labels = Tensor<Int32>(samples.map(\.label))
  return LabeledImage(data: imageTensor, label: labels)
}