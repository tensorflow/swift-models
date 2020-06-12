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

import Foundation
import TensorFlow

public struct COCODataset<Entropy: RandomNumberGenerator> {
  /// Type of the collection of non-collated batches.
  public typealias Batches = Slices<Sampling<[ObjectDetectionExample], ArraySlice<Int>>>
  /// The type of the training data, represented as a sequence of epochs, which
  /// are collection of batches.
  public typealias Training = LazyMapSequence<
    TrainingEpochs<[ObjectDetectionExample], Entropy>,
    LazyMapSequence<Batches, [ObjectDetectionExample]>
  >
  /// The type of the validation data, represented as a collection of batches.
  public typealias Validation = LazyMapSequence<Slices<[ObjectDetectionExample]>, [ObjectDetectionExample]>
  /// The training epochs.
  public let training: Training
  /// The validation batches.
  public let validation: Validation

  /// Creates an instance with `batchSize` on `device` using `remoteBinaryArchiveLocation`.
  ///
  /// - Parameters:
  ///   - training: The COCO metadata for the training data.
  ///   - validation: The COCO metadata for the validation data.
  ///   - includeMasks: Whether to include the segmentation masks when loading the dataset.
  ///   - batchSize: Number of images provided per batch.
  ///   - entropy: A source of randomness used to shuffle sample ordering.  It
  ///     will be stored in `self`, so if it is only pseudorandom and has value
  ///     semantics, the sequence of epochs is deterministic and not dependent
  ///     on other operations.
  ///   - device: The Device on which resulting Tensors from this dataset will be placed, as well
  ///     as where the latter stages of any conversion calculations will be performed.
  public init(
    training: COCO, validation: COCO, includeMasks: Bool, batchSize: Int,
    entropy: Entropy, device: Device
  ) {
    let trainingSamples = loadCOCOExamples(
      from: training,
      includeMasks: includeMasks,
      batchSize: batchSize)

    self.training = TrainingEpochs(samples: trainingSamples, batchSize: batchSize, entropy: entropy)
      .lazy.map { (batches: Batches) -> LazyMapSequence<Batches, [ObjectDetectionExample]> in
        return batches.lazy.map {
          makeBatch(samples: $0, device: device)
        }
      }

    let validationSamples = loadCOCOExamples(
      from: validation,
      includeMasks: includeMasks,
      batchSize: batchSize)

    self.validation = validationSamples.inBatches(of: batchSize).lazy.map {
      makeBatch(samples: $0, device: device)
    }
  }
}

func loadCOCOExamples(from coco: COCO, includeMasks: Bool, batchSize: Int)
    -> [ObjectDetectionExample]
{
    let images = coco.metadata["images"] as! [COCO.Image]
    let batchCount: Int = images.count / batchSize + 1
    let batches = Array(0..<batchCount)
    let examples: [[ObjectDetectionExample]] = batches.map { batchIdx in
        var examples: [ObjectDetectionExample] = []
        for i in 0..<batchSize {
            let idx = batchSize * batchIdx + i
            if idx < images.count {
                let img = images[idx]
                let example = loadCOCOExample(coco: coco, image: img, includeMasks: includeMasks)
                examples.append(example)
            }
        }
        return examples
    }
    let result = Array(examples.joined())
    assert(result.count == images.count)
    return result
}

func loadCOCOExample(coco: COCO, image: COCO.Image, includeMasks: Bool) -> ObjectDetectionExample {
    let imgDir = coco.imagesDirectory
    let imgW = image["width"] as! Int
    let imgH = image["height"] as! Int
    let imgFileName = image["file_name"] as! String
    var imgUrl: URL? = nil
    if imgDir != nil {
        let imgPath = imgDir!.appendingPathComponent(imgFileName).path
        imgUrl = URL(string: imgPath)!
    }
    let imgId = image["id"] as! Int
    let img = LazyImage(width: imgW, height: imgH, url: imgUrl)
    let annotations: [COCO.Annotation]
    if let anns = coco.imageToAnnotations[imgId] {
        annotations = anns
    } else {
        annotations = []
    }
    var objects: [LabeledObject] = []
    objects.reserveCapacity(annotations.count)
    for annotation in annotations {
        let bb = annotation["bbox"] as! [Double]
        let bbX = bb[0]
        let bbY = bb[1]
        let bbW = bb[2]
        let bbH = bb[3]
        let xMin = Float(bbX) / Float(imgW)
        let xMax = Float(bbX + bbW) / Float(imgW)
        let yMin = Float(bbY) / Float(imgH)
        let yMax = Float(bbY + bbH) / Float(imgH)
        let isCrowd: Int?
        if let iscrowd = annotation["iscrowd"] {
            isCrowd = iscrowd as? Int
        } else {
            isCrowd = nil
        }
        let area = Float(annotation["area"] as! Double)
        let classId = annotation["category_id"] as! Int
        let classInfo = coco.categories[classId]!
        let className = classInfo["name"] as! String
        let maskRLE: RLE?
        if includeMasks {
            maskRLE = coco.annotationToRLE(annotation)
        } else {
            maskRLE = nil
        }
        let object = LabeledObject(
            xMin: xMin, xMax: xMax,
            yMin: yMin, yMax: yMax,
            className: className, classId: classId,
            isCrowd: isCrowd, area: area, maskRLE: maskRLE)
        objects.append(object)
    }
    return ObjectDetectionExample(image: img, objects: objects)
}

fileprivate func makeBatch<BatchSamples: Collection>(
  samples: BatchSamples, device: Device
) -> [ObjectDetectionExample] where BatchSamples.Element == ObjectDetectionExample {
  return [ObjectDetectionExample](samples)
}
