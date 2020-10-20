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
// "Imagenette"
// Jeremy Howard
// https://github.com/fastai/imagenette

import Foundation
import ModelSupport
import TensorFlow

/// The three variants of Imagenette, determined by their source image size.
public enum ImagenetteSize {
  case full
  case resized160
  case resized320

  var suffix: String {
    switch self {
    case .full: return ""
    case .resized160: return "-160"
    case .resized320: return "-320"
    }
  }
}

public struct Imagenette<Entropy: RandomNumberGenerator> {
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
      .appendingPathComponent("Imagenette", isDirectory: true)
  ) {
    do {
      let trainingSamples = try loadImagenetteTrainingDirectory(
        inputSize: inputSize, localStorageDirectory: localStorageDirectory, base: "imagenette")

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

      let validationSamples = try loadImagenetteValidationDirectory(
        inputSize: inputSize, localStorageDirectory: localStorageDirectory, base: "imagenette")

      validation = validationSamples.inBatches(of: batchSize).lazy.map {
        makeImagenetteBatch(
          samples: $0, outputSize: outputSize, mean: mean, standardDeviation: standardDeviation,
          device: device)
      }
    } catch {
      fatalError("Could not load Imagenette dataset: \(error)")
    }
  }
}

extension Imagenette: ImageClassificationData where Entropy == SystemRandomNumberGenerator {
  /// Creates an instance with `batchSize`, using the SystemRandomNumberGenerator.
  public init(batchSize: Int, on device: Device = Device.default) {
    self.init(batchSize: batchSize, entropy: SystemRandomNumberGenerator(), device: device)
  }

  /// Creates an instance with `batchSize`, `inputSize`, and `outputSize`, using the
  /// SystemRandomNumberGenerator.
  public init(
    batchSize: Int, inputSize: ImagenetteSize, outputSize: Int, on device: Device = Device.default
  ) {
    self.init(
      batchSize: batchSize, entropy: SystemRandomNumberGenerator(), device: device,
      inputSize: inputSize, outputSize: outputSize)
  }
}

func downloadImagenetteIfNotPresent(to directory: URL, size: ImagenetteSize, base: String) {
  let downloadPath = directory.appendingPathComponent("\(base)\(size.suffix)").path
  let directoryExists = FileManager.default.fileExists(atPath: downloadPath)
  let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: downloadPath)
  let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)

  guard !directoryExists || directoryEmpty else { return }

  let location = URL(
    string: "https://s3.amazonaws.com/fast-ai-imageclas/\(base)\(size.suffix).tgz")!
  let _ = DatasetUtilities.downloadResource(
    filename: "\(base)\(size.suffix)", fileExtension: "tgz",
    remoteRoot: location.deletingLastPathComponent(), localStorageDirectory: directory)
}

func exploreImagenetteDirectory(
  named name: String, in directory: URL, inputSize: ImagenetteSize, base: String
) throws -> [URL] {
  downloadImagenetteIfNotPresent(to: directory, size: inputSize, base: base)
  let path = directory.appendingPathComponent("\(base)\(inputSize.suffix)/\(name)")
  let dirContents = try FileManager.default.contentsOfDirectory(
    at: path, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles])

  var urls: [URL] = []
  for directoryURL in dirContents {
    let subdirContents = try FileManager.default.contentsOfDirectory(
      at: directoryURL, includingPropertiesForKeys: [.isDirectoryKey],
      options: [.skipsHiddenFiles])
    urls += subdirContents
  }
  return urls
}

func parentLabel(url: URL) -> String {
  return url.deletingLastPathComponent().lastPathComponent
}

func createLabelDict(urls: [URL]) -> [String: Int] {
  let allLabels = urls.map(parentLabel)
  let labels = Array(Set(allLabels)).sorted()
  return Dictionary(uniqueKeysWithValues: labels.enumerated().map { ($0.element, $0.offset) })
}

func loadImagenetteDirectory(
  named name: String, in directory: URL, inputSize: ImagenetteSize, base: String,
  labelDict: [String: Int]? = nil
) throws -> [(file: URL, label: Int32)] {
  let urls = try exploreImagenetteDirectory(
    named: name, in: directory, inputSize: inputSize, base: base)
  let unwrappedLabelDict = labelDict ?? createLabelDict(urls: urls)
  return urls.lazy.map { (url: URL) -> (file: URL, label: Int32) in
    (file: url, label: Int32(unwrappedLabelDict[parentLabel(url: url)]!))
  }
}

func loadImagenetteTrainingDirectory(
  inputSize: ImagenetteSize, localStorageDirectory: URL, base: String,
  labelDict: [String: Int]? = nil
) throws
  -> [(file: URL, label: Int32)]
{
  return try loadImagenetteDirectory(
    named: "train", in: localStorageDirectory, inputSize: inputSize, base: base,
    labelDict: labelDict)
}

func loadImagenetteValidationDirectory(
  inputSize: ImagenetteSize, localStorageDirectory: URL, base: String,
  labelDict: [String: Int]? = nil
) throws
  -> [(file: URL, label: Int32)]
{
  return try loadImagenetteDirectory(
    named: "val", in: localStorageDirectory, inputSize: inputSize, base: base, labelDict: labelDict)
}

func makeImagenetteBatch<BatchSamples: Collection>(
  samples: BatchSamples, outputSize: Int, mean: Tensor<Float>?, standardDeviation: Tensor<Float>?,
  device: Device
) -> LabeledImage where BatchSamples.Element == (file: URL, label: Int32) {
  let images = samples.map(\.file).map { url -> Tensor<Float> in
    Image(contentsOf: url).resized(to: (outputSize, outputSize)).tensor
  }

  var imageTensor = Tensor(stacking: images)
  imageTensor = Tensor(copying: imageTensor, to: device)
  imageTensor /= 255.0

  if let mean = mean, let standardDeviation = standardDeviation {
    imageTensor = (imageTensor - mean) / standardDeviation
  }

  let labels = Tensor<Int32>(samples.map(\.label), on: device)
  return LabeledImage(data: imageTensor, label: labels)
}
