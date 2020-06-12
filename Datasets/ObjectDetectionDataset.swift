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
import ModelSupport
import TensorFlow

public struct LazyImage {
    public let width: Int
    public let height: Int
    public let url: URL?

    public init(width w: Int, height h: Int, url u: URL?) {
        self.width = w
        self.height = h
        self.url = u
    }

    public func tensor() -> Tensor<Float>? {
        if url != nil {
            return Image(jpeg: url!).tensor
        } else {
            return nil
        }
    }
}

public struct LabeledObject {
    public let xMin: Float
    public let xMax: Float
    public let yMin: Float
    public let yMax: Float
    public let className: String
    public let classId: Int
    public let isCrowd: Int?
    public let area: Float
    public let maskRLE: RLE?

    public init(
        xMin x0: Float, xMax x1: Float,
        yMin y0: Float, yMax y1: Float,
        className: String, classId: Int,
        isCrowd: Int?, area: Float, maskRLE: RLE?
    ) {
        self.xMin = x0
        self.xMax = x1
        self.yMin = y0
        self.yMax = y1
        self.className = className
        self.classId = classId
        self.isCrowd = isCrowd
        self.area = area
        self.maskRLE = maskRLE
    }
}

public struct ObjectDetectionExample: KeyPathIterable {
    public let image: LazyImage
    public let objects: [LabeledObject]

    public init(image: LazyImage, objects: [LabeledObject]) {
        self.image = image
        self.objects = objects
    }
}

/// Types whose elements represent an object detection dataset (with both
/// training and validation data).
public protocol ObjectDetectionData {
  /// The type of the training data, represented as a sequence of epochs, which
  /// are collection of batches.
  associatedtype Training: Sequence
  where Training.Element: Collection, Training.Element.Element == ObjectDetectionExample
  /// The type of the validation data, represented as a collection of batches.
  associatedtype Validation: Collection where Validation.Element == ObjectDetectionExample
  /// Creates an instance from a given `batchSize`.
  init(batchSize: Int, on device: Device)
  /// The `training` epochs.
  var training: Training { get }
  /// The `validation` batches.
  var validation: Validation { get }

  // The following is probably going to be necessary since we can't extract that
  // information from `Epochs` or `Batches`.
  /// The number of samples in the `training` set.
  //var trainingSampleCount: Int {get}
  /// The number of samples in the `validation` set.
  //var validationSampleCount: Int {get}
}
