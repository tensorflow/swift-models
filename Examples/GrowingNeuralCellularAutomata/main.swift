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

  @Option(help: "The number of training iterations.")
  var iterations = 8000

  @Option(help: "The minimum number of steps.")
  var minimumSteps = 64

  @Option(help: "The maximum number of steps.")
  var maximumSteps = 96

  @Option(help: "The number of steps to run through during inference.")
  var inferenceSteps = 96

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
    let paddedImage = inputImage.padded(forSizes: [(before: padding, after: padding), (before: padding, after: padding), (before: 0, after: 0)])
    let paddedImageBatch = paddedImage.broadcasted(to: [batchSize, paddedImage.shape[0], paddedImage.shape[1], paddedImage.shape[2]])
    
    // Initialize model, optimizer, and initial state.
    var initialState = Tensor(zerosLike: paddedImage).padded(forSizes: [(before: 0, after: 0), (before: 0, after: 0), (before: 0, after: stateChannels - 4)])
    initialState[initialState.shape[0] / 2][initialState.shape[1] / 2][3] = Tensor<Float>(1.0)
    
//    print("Initial state: \(initialState.scalars)")
    
    let initialBatch = initialState.broadcasted(to: [batchSize, initialState.shape[0], initialState.shape[1], initialState.shape[2]])
//    print("Initial batch: \(initialBatch.scalars)")
    var cellRule = CellRule(stateChannels: stateChannels)
    cellRule.move(to: device)
    var optimizer = Adam(for: cellRule, learningRate: 2e-3)
    optimizer = Adam(copying: optimizer, to: device)

    for iteration in 0..<iterations {
      let steps = Int.random(in: minimumSteps...maximumSteps)
      let (loss, ruleGradient) = valueWithGradient(at: cellRule) { cellRules -> Tensor<Float> in
        var state = initialBatch
        for _ in 0..<steps {
          state = cellRule(state)
        }
        // TODO: L2 normalization to parameter gradients.
        let rgbaComponents = state.slice(lowerBounds: [0, 0, 0, 0], sizes: [state.shape[0], state.shape[1], state.shape[2], 4])
        print("RGBA: \(rgbaComponents.shape), batch: \(paddedImageBatch.shape)")
        return meanSquaredError(predicted: rgbaComponents, expected: paddedImageBatch)
      }
      print("Rule gradient: \(ruleGradient)")
      optimizer.update(&cellRule, along: ruleGradient)
      LazyTensorBarrier()

      print("Iteration: \(iteration), loss: \(loss)")
      
      if (iteration % 2000) == 0 {
        optimizer.learningRate = optimizer.learningRate * 0.1
      }
    }
    
    // Perform inference and record the results.
    var state = initialState
    for step in 0..<inferenceSteps {
      state = cellRule(state)
      // TODO: Finish the color saving here.
      let colorComponents = state.slice(lowerBounds: [0, 0, 0], sizes: [state.shape[0], state.shape[1], 4])
      // TODO: Add RGBA saving with PNG format here.
      let filename = String(format: "step%03d", step)
      try saveImage(colorComponents, shape: (colorComponents.shape[0], colorComponents.shape[1]), format: .rgb, directory: "output", name: filename)
    }
  }
}

GrowingNeuralCellularAutomata.main()
