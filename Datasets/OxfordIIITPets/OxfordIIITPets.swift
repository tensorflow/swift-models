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

// Original Source
// "The Oxford-IIIT Pet Dataset"
// Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman and C. V. Jawahar
// https://www.robots.ox.ac.uk/~vgg/data/pets/

import Foundation
import ModelSupport
import TensorFlow

public struct OxfordIIITPets<Entropy: RandomNumberGenerator> {
  /// Type of the collection of non-collated batches.
  public typealias Batches = Slices<Sampling<[(file: URL, annotation: URL)], ArraySlice<Int>>>
  /// The type of the training data, represented as a sequence of epochs, which
  /// are collection of batches.
  public typealias Training = LazyMapSequence<
    TrainingEpochs<[(file: URL, annotation: URL)], Entropy>,
    LazyMapSequence<Batches, SegmentedImage>
  >
  /// The type of the validation data, represented as a collection of batches.
  public typealias Validation = LazyMapSequence<
    Slices<[(file: URL, annotation: URL)]>, LabeledImage
  >
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
      batchSize: batchSize, entropy: entropy, device: device, imageSize: 224)
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
  ///   - imageSize: The square width and height of the images returned from this dataset.
  ///   - localStorageDirectory: Where to place the downloaded and unarchived dataset.
  public init(
    batchSize: Int, entropy: Entropy, device: Device, imageSize: Int,
    localStorageDirectory: URL = DatasetUtilities.defaultDirectory
      .appendingPathComponent("OxfordIIITPets", isDirectory: true)
  ) {
    do {
      let trainingSamples = try loadOxfordIITPetsTraining(
        localStorageDirectory: localStorageDirectory)

      training = TrainingEpochs(samples: trainingSamples, batchSize: batchSize, entropy: entropy)
        .lazy.map { (batches: Batches) -> LazyMapSequence<Batches, LabeledImage> in
          return batches.lazy.map {
            makeBatch(samples: $0, imageSize: imageSize, device: device)
          }
        }

      let validationSamples = try loadOxfordIITPetsTraining(
        localStorageDirectory: localStorageDirectory)

      validation = validationSamples.inBatches(of: batchSize).lazy.map {
        makeBatch(samples: $0, imageSize: imageSize, device: device)
      }
    } catch {
      fatalError("Could not load the Oxford IIIT Pets dataset: \(error)")
    }
  }
}

func downloadOxfordIIITPetsIfNotPresent(to directory: URL) {
  let downloadPath = directory.appendingPathComponent("images", isDirectory: true).path
  let directoryExists = FileManager.default.fileExists(atPath: downloadPath)
  let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: downloadPath)
  let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)

  guard !directoryExists || directoryEmpty else { return }

  let remoteRoot = URL(string: "https://www.robots.ox.ac.uk/~vgg/data/pets/data/")!

  let _ = DatasetUtilities.downloadResource(
    filename: "images", fileExtension: "tar.gz",
    remoteRoot: remoteRoot, localStorageDirectory: directory
  )

  let _ = DatasetUtilities.downloadResource(
    filename: "annotations", fileExtension: "tar.gz",
    remoteRoot: remoteRoot, localStorageDirectory: directory
  )
}

func loadOxfordIIITPets(filename: String, in directory: URL) throws -> [(
  file: URL, annotation: URL
)] {
  downloadOxfordIIITPetsIfNotPresent(to: directory)
  let imageURLs = getImageURLs(filename: filename, directory: directory)
  return imageURLs.lazy.map { (imageURL: URL) -> (file: URL, annotation: URL) in
    (file: imageURL, annotation: makeAnnotationURL(imageURL: imageURL, directory: directory))
  }
}

func makeAnnotationURL(imageURL: URL, directory: URL) -> URL {
  let filename = imageURL.deletingPathExtension().lastPathComponent
  return directory.appendingPathComponent("annotations/trimaps/\(filename).png")
}

func getImageURLs(filename: String, directory: URL) -> [URL] {
  let filePath = directory.appendingPathComponent("annotations/\(filename)")
  let imagesRootDirectory = directory.appendingPathComponent("images", isDirectory: true)
  let fileContents = try? String(contentsOf: filePath)
  let imageDetails = fileContents!.split(separator: "\n")
  return imageDetails.map {
    let imagename = String($0[..<$0.firstIndex(of: " ")!])
    return imagesRootDirectory.appendingPathComponent("\(imagename).jpg")
  }
}

func loadOxfordIITPetsTraining(localStorageDirectory: URL) throws -> [(file: URL, annotation: URL)]
{
  return try loadOxfordIIITPets(
    filename: "trainval.txt", in: localStorageDirectory)
}

func loadOxfordIIITPetsValidation(localStorageDirectory: URL) throws -> [(
  file: URL, annotation: URL
)] {
  return try loadOxfordIIITPets(
    filename: "test.txt", in: localStorageDirectory)
}

fileprivate func makeBatch<BatchSamples: Collection>(
  samples: BatchSamples, imageSize: Int, device: Device
) -> SegmentedImage where BatchSamples.Element == (file: URL, annotation: URL) {
  let images = samples.map(\.file).map { url -> Tensor<Float> in
    Image(jpeg: url).resized(to: (imageSize, imageSize)).tensor[0..., 0..., 0..<3]
  }

  var imageTensor = Tensor(stacking: images)
  imageTensor = Tensor(copying: imageTensor, to: device)
  imageTensor /= 255.0

  let annotations = samples.map(\.annotation).map { url -> Tensor<Int32> in
    Tensor<Int32>(
      Image(jpeg: url).resized(to: (imageSize, imageSize)).tensor[0..., 0..., 0...0] - 1)
  }
  var annotationTensor = Tensor(stacking: annotations)
  annotationTensor = Tensor(copying: annotationTensor, to: device)

  return SegmentedImage(data: imageTensor, label: annotationTensor)
}
