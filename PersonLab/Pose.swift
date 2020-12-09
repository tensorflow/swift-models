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

import TensorFlow

struct Keypoint {
  var y: Float
  var x: Float
  let index: KeypointIndex
  let score: Float

  init(
    heatmapY: Int, heatmapX: Int, index: Int, score: Float, offsets: CPUTensor<Float>,
    outputStride: Int
  ) {
    self.y = Float(heatmapY) * Float(outputStride) + offsets[heatmapY, heatmapX, index]
    self.x =
      Float(heatmapX) * Float(outputStride)
      + offsets[heatmapY, heatmapX, index + KeypointIndex.allCases.count]
    self.index = KeypointIndex(rawValue: index)!
    self.score = score
  }

  init(y: Float, x: Float, index: KeypointIndex, score: Float) {
    self.y = y
    self.x = x
    self.index = index
    self.score = score
  }

  func isWithinRadiusOfCorrespondingKeypoints(in poses: [Pose], radius: Float) -> Bool {
    return poses.contains { pose in
      let correspondingKeypoint = pose.getKeypoint(self.index)!
      let dy = correspondingKeypoint.y - self.y
      let dx = correspondingKeypoint.x - self.x
      let squaredDistance = dy * dy + dx * dx
      return squaredDistance <= radius * radius
    }
  }
}

enum KeypointIndex: Int, CaseIterable {
  case nose = 0
  case leftEye
  case rightEye
  case leftEar
  case rightEar
  case leftShoulder
  case rightShoulder
  case leftElbow
  case rightElbow
  case leftWrist
  case rightWrist
  case leftHip
  case rightHip
  case leftKnee
  case rightKnee
  case leftAnkle
  case rightAnkle
}

enum Direction { case fwd, bwd }

func getNextKeypointIndexAndDirection(_ keypointId: KeypointIndex) -> [(KeypointIndex, Direction)] {
  switch keypointId {
  case .nose:
    return [(.leftEye, .fwd), (.rightEye, .fwd), (.leftShoulder, .fwd), (.rightShoulder, .fwd)]
  case .leftEye: return [(.nose, .bwd), (.leftEar, .fwd)]
  case .rightEye: return [(.nose, .bwd), (.rightEar, .fwd)]
  case .leftEar: return [(.leftEye, .bwd)]
  case .rightEar: return [(.rightEye, .bwd)]
  case .leftShoulder: return [(.leftHip, .fwd), (.leftElbow, .fwd), (.nose, .bwd)]
  case .rightShoulder: return [(.rightHip, .fwd), (.rightElbow, .fwd), (.nose, .bwd)]
  case .leftElbow: return [(.leftWrist, .fwd), (.leftShoulder, .bwd)]
  case .rightElbow: return [(.rightWrist, .fwd), (.rightShoulder, .bwd)]
  case .leftWrist: return [(.leftElbow, .bwd)]
  case .rightWrist: return [(.rightElbow, .bwd)]
  case .leftHip: return [(.leftKnee, .fwd), (.leftShoulder, .bwd)]
  case .rightHip: return [(.rightKnee, .fwd), (.rightShoulder, .bwd)]
  case .leftKnee: return [(.leftAnkle, .fwd), (.leftHip, .bwd)]
  case .rightKnee: return [(.rightAnkle, .fwd), (.rightHip, .bwd)]
  case .leftAnkle: return [(.leftKnee, .bwd)]
  case .rightAnkle: return [(.rightKnee, .bwd)]
  }
}

/// Maps a pair of keypoint indexes to the appropiate index to be used
/// in the displacement forward and backward tensors.
let keypointPairToDisplacementIndexMap: [Set<KeypointIndex>: Int] = [
  Set([.nose, .leftEye]): 0,
  Set([.leftEye, .leftEar]): 1,
  Set([.nose, .rightEye]): 2,
  Set([.rightEye, .rightEar]): 3,
  Set([.nose, .leftShoulder]): 4,
  Set([.leftShoulder, .leftElbow]): 5,
  Set([.leftElbow, .leftWrist]): 6,
  Set([.leftShoulder, .leftHip]): 7,
  Set([.leftHip, .leftKnee]): 8,
  Set([.leftKnee, .leftAnkle]): 9,
  Set([.nose, .rightShoulder]): 10,
  Set([.rightShoulder, .rightElbow]): 11,
  Set([.rightElbow, .rightWrist]): 12,
  Set([.rightShoulder, .rightHip]): 13,
  Set([.rightHip, .rightKnee]): 14,
  Set([.rightKnee, .rightAnkle]): 15,
]

public struct Pose {
  var keypoints: [Keypoint?] = Array(repeating: nil, count: KeypointIndex.allCases.count)
  var resolution: (height: Int, width: Int)

  mutating func add(_ keypoint: Keypoint) {
    keypoints[keypoint.index.rawValue] = keypoint
  }

  func getKeypoint(_ index: KeypointIndex) -> Keypoint? {
    return keypoints[index.rawValue]
  }

  mutating func rescale(to newResolution: (height: Int, width: Int)) {
    for i in 0..<keypoints.count {
      if var k = keypoints[i] {
        k.y *= Float(newResolution.height) / Float(resolution.height)
        k.x *= Float(newResolution.width) / Float(resolution.width)
        self.keypoints[i] = k
      }
    }
    self.resolution = newResolution
  }
}

extension Pose: CustomStringConvertible {
  public var description: String {
    var description = ""
    for keypoint in keypoints {
      description.append(
        "\(keypoint!.index) - \(keypoint!.score) | \(keypoint!.y) - \(keypoint!.x)\n")
    }
    return description
  }
}
