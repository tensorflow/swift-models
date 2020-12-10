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

// ImageNet data source:
// http://www.image-net.org/challenges/LSVRC/2012/index#cite
// "ImageNet Large Scale Visual Recognition Challenge"
// https://arxiv.org/abs/1409.0575

// Post-processing applied:
// 1) Download ImageNet files (eg ILSVRC2012_img_train.tar (A), ILSVRC2012_img_val.tar (B))
// A) untar tar file to produce 1000 tar files in a folder called 'train':
//    untar each + create directories:
//    > mkdir n01440764; tar -xvf n01440764.tar -C n01440764; rm n01440764.tar
// B) untar 50k images to a folder called `val`:
//    move images to labeled subfolders using pytorch script:
//    https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
//    remove blacklisted validation images:
//    https://raw.githubusercontent.com/fastai/imagenet-fast/master/imagenet_nv/blacklist.sh
// 2) create imagenet.tgz: tar -czvf imagenet.tgz train val

import Foundation
import ModelSupport
import TensorFlow

public struct ImageNet<Entropy: RandomNumberGenerator> {
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
      batchSize: batchSize, entropy: entropy, device: device,
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
  ///   - outputSize: The square width and height of the images returned from this dataset.
  ///   - localStorageDirectory: Where to place the downloaded and unarchived dataset.
  public init(
    batchSize: Int, entropy: Entropy, device: Device,
    outputSize: Int,
    localStorageDirectory: URL = DatasetUtilities.defaultDirectory
      .appendingPathComponent("ImageNet", isDirectory: true)
  ) {
    do {
      let trainingSamples = try loadImageNetTrainingDirectory(
         localStorageDirectory: localStorageDirectory, base: "imagenet")

      let mean = Tensor<Float>([0.485, 0.456, 0.406], on: device)
      let standardDeviation = Tensor<Float>([0.229, 0.224, 0.225], on: device)

      training = TrainingEpochs(samples: trainingSamples, batchSize: batchSize, entropy: entropy)
        .lazy.map { (batches: Batches) -> LazyMapSequence<Batches, LabeledImage> in
          return batches.lazy.map {
            makeImageNetBatch(
              samples: $0, outputSize: outputSize, mean: mean, standardDeviation: standardDeviation,
              device: device)
          }
        }

      let validationSamples = try loadImageNetValidationDirectory(localStorageDirectory: localStorageDirectory, base: "imagenet")

      validation = validationSamples.inBatches(of: batchSize).lazy.map {
        makeImageNetBatch(
          samples: $0, outputSize: outputSize, mean: mean, standardDeviation: standardDeviation,
          device: device)
      }
    } catch {
      fatalError("Could not load ImageNet dataset: \(error)")
    }
  }
}

extension ImageNet: ImageClassificationData where Entropy == SystemRandomNumberGenerator {
  /// Creates an instance with `batchSize`, using the SystemRandomNumberGenerator.
  public init(batchSize: Int, on device: Device = Device.default) {
    self.init(batchSize: batchSize, entropy: SystemRandomNumberGenerator(), device: device)
  }

  /// Creates an instance with `batchSize` and `outputSize`, using the
  /// SystemRandomNumberGenerator.
  public init(
    batchSize: Int, outputSize: Int, on device: Device = Device.default
  ) {
    self.init(
      batchSize: batchSize, entropy: SystemRandomNumberGenerator(), device: device,
     outputSize: outputSize)
  }
}

func downloadImageNetIfNotPresent(to directory: URL, base: String) {
  let downloadPath = directory.appendingPathComponent("\(base)").path
  let directoryExists = FileManager.default.fileExists(atPath: downloadPath)
  let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: downloadPath)
  let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)

  guard !directoryExists || directoryEmpty else { return }

  // this approach tries to work in memory --> ~150GB in-memory download --> hits swap  -> stream to file instead?
  // let location = URL(
  //   string: "https://REMOTE-SERVER/imagenet/imagenet.tgz")!
  // let _ = DatasetUtilities.downloadResource(
  //   filename: "\(base)\(size.suffix)", fileExtension: "tgz",
  //   remoteRoot: location.deletingLastPathComponent(), localStorageDirectory: directory)
  // END ORIGINAL CODE

  print("Assuming you have downloaded ImageNet to '/tmp/imagenet.tgz', starting extract.")
  extractArchive(at: URL(string:"/tmp/imagenet.tgz")!, to: URL(string: downloadPath)!,
               fileExtension: "tgz", deleteArchiveWhenDone: false)
  print("Done extracting'.")
}

func exploreImageNetDirectory(
  named name: String, in directory: URL, base: String
) throws -> [URL] {
  downloadImageNetIfNotPresent(to: directory, base: base)
  let path = directory.appendingPathComponent("\(base)/\(name)")
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

func loadImageNetDirectory(
  named name: String, in directory: URL, base: String,
  labelDict: [String: Int]? = nil
) throws -> [(file: URL, label: Int32)] {
  let urls = try exploreImageNetDirectory(
    named: name, in: directory, base: base)
  let unwrappedLabelDict = labelDict ?? createLabelDict(urls: urls)
  return urls.lazy.map { (url: URL) -> (file: URL, label: Int32) in
    (file: url, label: Int32(unwrappedLabelDict[parentLabel(url: url)]!))
  }
}

func loadImageNetTrainingDirectory(
  localStorageDirectory: URL, base: String,
  labelDict: [String: Int]? = nil
) throws
  -> [(file: URL, label: Int32)]
{
  return try loadImageNetDirectory(
    named: "train", in: localStorageDirectory, base: base,
    labelDict: labelDict)
}

func loadImageNetValidationDirectory(
  localStorageDirectory: URL, base: String,
  labelDict: [String: Int]? = nil
) throws
  -> [(file: URL, label: Int32)]
{
  return try loadImageNetDirectory(
    named: "val", in: localStorageDirectory, base: base, labelDict: labelDict)
}

func applyImageNetDataAugmentation(image: Image) -> Tensor<Float> {
    // using the tensorflow imagenet demo from mlperf as reference:
    // https://github.com/mlcommons/training/blob/4f97c909f3aeaa3351da473d12eba461ace0be76/image_classification/tensorflow/official/resnet/imagenet_preprocessing.py#L94
    let imageData = image.tensor
    let (height, width, channels) = (imageData.shape[0], imageData.shape[1], imageData.shape[2])

    let imageSize = Tensor([Int32(height), Int32(width), Int32(channels)])
    let bboxes = Tensor<Float>(shape: [1, 1, 4], scalars: [0.0, 0.0, 1.0, 1.0])

    // the default values for this op internally are the imagenet settings
    let randomCrop = _Raw.sampleDistortedBoundingBox(imageSize: imageSize, boundingBoxes: bboxes)
    let offsets = randomCrop.begin
    let targets = randomCrop.size

    // we manually convert to normalized coordinates
    let offsetY = Float(offsets[0].scalar!) / Float(height)
    let offsetX = Float(offsets[1].scalar!) / Float(width)
    let targetY = Float(targets[0].scalar!) / Float(height) + offsetY
    let targetX = Float(targets[1].scalar!) / Float(width) + offsetX

    var cropped = Tensor<Float>([offsetY, offsetX, targetY, targetX])
    // we add a random flip here by swapping the x coordinates
    if Bool.random() {
      cropped = Tensor<Float>([offsetY, targetX, targetY, offsetX])
    }

    let imageBroadcast = imageData.reshaped(to: [1, height, width, channels])
    let bboxBroadcast = cropped.reshaped(to: [1, 4])

    let croppedImage = _Raw.cropAndResize(image: imageBroadcast, boxes: bboxBroadcast,
      boxInd: [0], cropSize: [224, 224])
    return croppedImage.reshaped(to: [224, 224, 3])
}

func makeImageNetBatch<BatchSamples: Collection>(
  samples: BatchSamples, outputSize: Int, mean: Tensor<Float>?, standardDeviation: Tensor<Float>?,
  device: Device
) -> LabeledImage where BatchSamples.Element == (file: URL, label: Int32) {
  let images = samples.map(\.file).map { url -> Tensor<Float> in
    if url.absoluteString.range(of: "n02105855_2933.JPEG") != nil {
      // this is a png saved as a jpeg, we manually strip an extra alpha channel to start
      let image = Image(contentsOf: url).tensor.slice(lowerBounds: [0, 0, 0], sizes: [189, 213, 3])
      let colorOnlyImage = Image(image)
      return applyImageNetDataAugmentation(image: colorOnlyImage)
    } else {
      let image = Image(contentsOf: url)
      return applyImageNetDataAugmentation(image: image)
    }
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
