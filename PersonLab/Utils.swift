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

public struct Config {
  let printProfilingData: Bool
  var checkpointPath = URL(
    string:
      "https://github.com/tryolabs/swift-models/releases/download/PersonlabDemo/personlabCheckpoint.zip"
  )!
  let inputImageSize = (height: 241, width: 289)

  // Decoder
  let outputStride = 16
  let poseScoreThreshold: Float = 0.15
  let keypointScoreThreshold: Float = 0.1
  let nmsRadius: Float = 20.0
  let keypointLocalMaximumRadius = 1
}

extension CheckpointReader {
  func load(from name: String) -> Tensor<Float> {
    return Tensor(self.loadTensor(named: "MobilenetV1/\(name)"))
  }
}

func draw(_ pose: Pose, on imageTensor: inout Tensor<Float>) {
  var pose = pose
  pose.rescale(to: (height: imageTensor.shape[0], width: imageTensor.shape[1]))

  func recursivellyDrawNextKeypoint(
    after previousKeypoint: Keypoint, into imageTensor: inout Tensor<Float>
  ) {
    for (nextKeypointIndex, direction) in getNextKeypointIndexAndDirection(previousKeypoint.index) {
      if direction == .fwd {
        if let nextKeypoint = pose.getKeypoint(nextKeypointIndex) {
          drawLine(
            on: &imageTensor,
            from: (Int(previousKeypoint.x), Int(previousKeypoint.y)),
            to: (Int(nextKeypoint.x), Int(nextKeypoint.y))
          )
          recursivellyDrawNextKeypoint(after: nextKeypoint, into: &imageTensor)
        }
      }
    }
  }

  recursivellyDrawNextKeypoint(after: pose.getKeypoint(.nose)!, into: &imageTensor)
}

/// Used as an ad-hoc "hash" for tensor checking when copying the backbone from
/// our Python Tensorflow 1.5 version
func hash(_ tensor: Tensor<Float>) {
  print(
    "[\(tensor.flattened().sum()), \(tensor[0, 0, 0]) \(tensor[0, -1, 1]), \(tensor[0, 1, 0]), \(tensor[0, -1, -1])]"
  )
}

/// Wrapper for Tensor which allows several order of magnitude faster subscript access,
/// as it avoids unnecesary GPU->CPU copies on each access.
struct CPUTensor<T: TensorFlowScalar> {
  private var flattenedTensor: [T]
  var shape: TensorShape

  init(_ tensor: Tensor<T>) {
    self.flattenedTensor = tensor.scalars
    self.shape = tensor.shape
  }

  subscript(indexes: Int...) -> T {
    var oneDimensionalIndex = 0
    for i in 1..<shape.count {
      oneDimensionalIndex += indexes[i - 1] * shape[i...].reduce(1, *)
    }
    // Last dimension doesn't have multipliers.
    oneDimensionalIndex += indexes.last!
    return flattenedTensor[oneDimensionalIndex]
  }
}
