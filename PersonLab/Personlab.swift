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

import Checkpoints
import Foundation
import ModelSupport
import TensorFlow

public struct PersonLab {
  let config: Config
  let ckpt: CheckpointReader
  let backbone: MobileNetLikeBackbone
  let personlabHeads: PersonlabHeads

  public init(_ config: Config) {
    self.config = config
    do {
      self.ckpt = try CheckpointReader(
        checkpointLocation: config.checkpointPath, modelName: "Personlab"
      )
    } catch {
      print("Error loading checkpoint file: \(config.checkpointPath)")
      print(error)
      exit(0)
    }
    self.backbone = MobileNetLikeBackbone(checkpoint: ckpt)
    self.personlabHeads = PersonlabHeads(checkpoint: ckpt)
  }

  public func callAsFunction(_ inputImage: Image) -> [Pose] {
    let startTime = Date()

    let resizedImage = inputImage.resized(to: config.inputImageSize)
    let normalizedImageTensor = resizedImage.tensor * (2.0 / 255.0) - 1.0
    let batchedNormalizedImagesTensor = normalizedImageTensor.expandingShape(at: 0)
    let preprocessingTime = Date()

    let convnetResults = personlabHeads(backbone(batchedNormalizedImagesTensor))
    let convnetTime = Date()

    let poseDecoder = PoseDecoder(for: convnetResults, with: self.config)
    let poses = poseDecoder.decode()
    let decoderTime = Date()

    if self.config.printProfilingData {
      print(
        String(
          format: "Preprocessing: %.2f ms", preprocessingTime.timeIntervalSince(startTime) * 1000),
        "|",
        String(
          format: "Backbone: %.2f ms", convnetTime.timeIntervalSince(preprocessingTime) * 1000),
        "|",
        String(format: "Decoder: %.2f ms", decoderTime.timeIntervalSince(convnetTime) * 1000)
      )
    }

    return poses
  }

}
