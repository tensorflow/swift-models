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

// Original Source
// "Learning and Inferring Motion Patterns using Parametric Segmental Switching Linear Dynamic Systems"
// Sang Min Oh, James M. Rehg, Tucker Balch, Frank Dellaert
// International Journal of Computer Vision (IJCV) Special Issue on Learning for Vision, May 2008. Vol.77(1-3). Pages 103-124.
// https://www.cc.gatech.edu/~borg/ijcv_psslds/

import Foundation
import ModelSupport
import TensorFlow

/// An ordered collection of frames from a honey bee dance video.
public struct HoneyBeeDanceFrames: RandomAccessCollection {
  public let directory: URL
  public let frameCount: Int

  public enum Error: Swift.Error {
    case initError(String)
  }

  /// Creates a `HoneyBeeDanceFrames` from the data in the given `directory`.
  ///
  /// The directory must contain:
  /// - A file named "index.txt" whose first line is the total number of frames.
  /// - Frames named "frame1.png", "frame2.png", etc.
  public init(directory: URL) throws {
    let indexFile = directory.appendingPathComponent("index.txt")
    let index = try String(contentsOf: indexFile)
    guard let indexLine = index.split(separator: "\n").first else {
      throw Error.initError("index.txt empty")
    }
    guard let frameCount = Int(indexLine) else {
      throw Error.initError("index.txt first line is not a number")
    }
    self.directory = directory
    self.frameCount = frameCount
  }

  public var startIndex: Int { 0 }
  public var endIndex: Int { frameCount }

  public func index(before i: Int) -> Int { i - 1 }
  public func index(after i: Int) -> Int { i + 1 }

  public subscript(index: Int) -> Image {
    return Image(jpeg: directory.appendingPathComponent("frame\(index + 1).png"))
  }
}

public struct HoneyBeeDanceSegmentation {
  public let images: Tensor<Float>
  public let annotations: Tensor<Bool>

  public enum Error: Swift.Error {
    case initError(String)
  }

  public init(directory: URL) throws {
    var images: [Tensor<Float>] = []
    var annotations: [Tensor<Bool>] = []

    let trainFile = directory.appendingPathComponent("train.txt")
    let train = try String(contentsOf: trainFile)
    for line in train.split(separator: "\n") {
      if line.count == 0 { continue }
      let cols = line.split(separator: " ")
      guard cols.count == 2 else {
        print(cols)
        throw Error .initError("unexpected number of columns")
      }

      let image = Image(jpeg: directory.appendingPathComponent(String(cols[0]))).tensor
      let annotation = Image(jpeg: directory.appendingPathComponent(String(cols[1]))).tensor

      guard image.shape.count == 3 else {
        throw Error.initError("unexpected image rank")
      }
      guard annotation.shape.count == 3 else {
        throw Error.initError("unexpected annotation rank")
      }
      guard image.shape[0..<2] == annotation.shape[0..<2] else {
        throw Error.initError("differing image an annotation sizes \(image.shape) \(annotation.shape)")
      }

      let tileSize = 120

      for i in 0..<(image.shape[0] / tileSize) {
        for j in 0..<(image.shape[1] / tileSize) {
          let x0 = Int32(tileSize * i)
          let y0 = Int32(tileSize * j)
          images.append(
            image.slice(
              lowerBounds: Tensor([x0, y0, 0]),
              sizes: Tensor([Int32(tileSize), Int32(tileSize), Int32(image.shape[2])])))

          let annotation = annotation.slice(
            lowerBounds: Tensor([x0, y0, 0]),
            sizes: Tensor([Int32(tileSize), Int32(tileSize), 1]))
          annotations.append(annotation .== Tensor(255))
        }
      }
    }

    self.images = Tensor(stacking: images)
    self.annotations = Tensor(stacking: annotations)
  }
}
