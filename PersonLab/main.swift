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

import ArgumentParser
import Foundation
import ModelSupport
import TensorFlow

struct Inference: ParsableCommand {
  static var configuration = CommandConfiguration(
    commandName: "personlab",
    abstract: """
      Runs human pose estimation on a local image file.
      """
  )

  @Argument(help: "Path to local image to run pose estimation on")
  var imagePath: String

  @Option(name: .shortAndLong, help: "Path to checkpoint directory")
  var checkpointPath: String?

  @Flag(name: .shortAndLong, help: "Print profiling data")
  var profiling = false

  func run() {
    Context.local.learningPhase = .inference
    var config = Config(printProfilingData: profiling)
    if checkpointPath != nil {
      config.checkpointPath = URL(fileURLWithPath: checkpointPath!)
    }
    let model = PersonLab(config)

    let fileManager = FileManager()
    if !fileManager.fileExists(atPath: imagePath) {
      print("No image found at path: \(imagePath)")
      return
    }
    let image = Image(contentsOf: URL(fileURLWithPath: imagePath))

    var poses = [Pose]()
    if profiling {
      print("Running model 10 times to see how inference time changes.")
      for _ in 1...10 {
        poses = model(image)
      }
    } else {
      poses = model(image)
    }

    var drawnTensor = image.tensor
    for pose in poses {
      draw(pose, on: &drawnTensor)
    }
    do {
      try drawnTensor.saveImage(directory: "./", name: "out")
      print("Output image saved to 'out.jpg'")
    } catch {
      print("Error during final image output: \(error).")
    }
  }
}

Inference.main()
