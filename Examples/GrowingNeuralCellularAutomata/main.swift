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

struct GrowingNeuralCellularAutomata: ParsableCommand {
  static var configuration = CommandConfiguration(
    commandName: "GrowingNeuralCellularAutomata",
    abstract: "Neural cellular automata with rules trained to grow in the shape of images.",
    subcommands: [])
  
  @Flag(help: "Use eager backend.")
  var eager = false

  @Flag(help: "Use X10 backend.")
  var x10 = false
  
  @Option(help: "The height and width to use when resizing the input image.")
  var imageSize = 40

  @Option(help: "The padding to add around the input image after resizing.")
  var padding = 16
  
  @Option(help: "The number of state channels for each cell.")
  var stateChannels = 16

  @Option(help: "The batch size during training.")
  var batchSize = 8

  @Option(help: "The fraction of cells to fire at each update.")
  var cellFireRate: Float = 0.5

  @Option(help: "The pool size during training.")
  var poolSize = 1024

  @Option(help: "The image to use as a target.")
  var image: String

  func validate() throws {
    guard !(eager && x10) else {
      throw ValidationError(
        "Can't specify both --eager and --x10 backends.")
    }
    
    guard stateChannels > 4 else {
      throw ValidationError(
        "Must have at least 4 channels to support RGBA values.")
    }
  }
  
  func run() throws {
    // TODO: Remove this workaround to prevent excessive TF memory growth when fixed upstream.
    let _ = _ExecutionContext.global
    
    // Set up the backend.
    let device: Device
    if x10 {
      device = Device.defaultXLA
    } else {
      device = Device.defaultTFEager
    }
    
    // Load and pad the target image to evolve towards.
    let hostInputImage = Image(jpeg: URL(fileURLWithPath: image)).resized(to: (imageSize, imageSize)).tensor
    let inputImage = Tensor(copying: hostInputImage, to: device)
    print("Input image: \(inputImage.shape)")
    let paddedImage = inputImage.padded(forSizes: [(before: padding, after: padding), (before: padding, after: padding), (before: 0, after: 0)])
    print("Padded image: \(paddedImage.shape)")
    
    var cellRules = CellRules(stateChannels: stateChannels)
    
    // Start training loop
    
    // Seed initial image
    // Run for 64-96 steps
    // Perform loss
    // Calculate and apply gradient via Adam
    // LR = 2e-3
    // Piecewise constant LR decay (2000, [lr, lr*0.1])
    
    // Seed inference image
    // Perform inference
    // Capture images at various stages
  }
}

GrowingNeuralCellularAutomata.main()
