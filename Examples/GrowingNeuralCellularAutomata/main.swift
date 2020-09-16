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

  @Option(help: "The image to use as a target.")
  var image: String

  @Option(help: "The height and width to use when resizing the input image.")
  var imageSize = 40

  @Option(help: "The number of training iterations.")
  var iterations = 8000

  @Option(help: "The number of steps to run through during inference.")
  var inferenceSteps = 96

  @Option(help: "The padding to add around the input image after resizing.")
  var padding = 16

  @Option(help: "The number of state channels for each cell.")
  var stateChannels = 16

  @Option(help: "The batch size during training.")
  var batchSize = 8

  @Option(help: "The fraction of cells to fire at each update.")
  var cellFireRate: Float = 0.5

  @Option(help: "The minimum number of steps.")
  var minimumSteps = 64

  @Option(help: "The maximum number of steps.")
  var maximumSteps = 96

  @Flag(help: "Whether to use a sample pool.")
  var useSamplePool = false

  @Option(help: "The pool size during training.")
  var poolSize = 1024

  @Option(help: "The number of samples to damage in each batch.")
  var damagedSamples = 0

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
  func recordGrowth(
    seed: Tensor<Float>, rule: CellRule, steps: Int, directory: String, filename: String
  ) throws -> Tensor<Float> {
    var state = seed
    var states: [Tensor<Float>] = []
    LazyTensorBarrier()
    for _ in 0..<steps {
      state = rule(state)
      let sampledState = state[0]
      LazyTensorBarrier()
      states.append(sampledState.colorComponents * 255.0)
    }
    try saveAnimatedImage(states, delay: 1, directory: directory, name: filename)
    return state
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
    let hostInputImage = Image(contentsOf: URL(fileURLWithPath: image))
    let resizedHostInputImage = hostInputImage.resized(to: (imageSize, imageSize))
    let inputImage = Tensor(copying: resizedHostInputImage.tensor, to: device) / 255.0
    let paddedImage = inputImage.padded(forSizes: [
      (before: padding, after: padding), (before: padding, after: padding), (before: 0, after: 0),
    ])
    let paddedImageBatch = paddedImage.broadcasted(to: [
      batchSize, paddedImage.shape[0], paddedImage.shape[1], paddedImage.shape[2],
    ])

    try saveImage(
      paddedImage * 255.0, colorspace: .rgba, directory: "output", name: "targetimage", format: .png)

    // Initialize model, optimizer, and initial state.
    var initialState = Tensor(zerosLike: paddedImage).padded(forSizes: [
      (before: 0, after: 0), (before: 0, after: 0), (before: 0, after: stateChannels - 4),
    ])
    initialState[initialState.shape[0] / 2][initialState.shape[1] / 2][3] = Tensor<Float>(1.0, on: device)
    var initialBatch = initialState.broadcasted(to: [
      batchSize, initialState.shape[0], initialState.shape[1], initialState.shape[2],
    ])
    
    // TODO: Make this optional when we can differentiate through optionals.
    var samplePool: SamplePool
    if useSamplePool {
      samplePool = SamplePool(initialState: initialState, size: poolSize)
    } else {
      samplePool = SamplePool(initialState: initialState, size: 0)
    }
    
    var cellRule = CellRule(stateChannels: stateChannels, fireRate: cellFireRate)
    cellRule.move(to: device)
    var optimizer = Adam(for: cellRule, learningRate: 2e-3)
    optimizer = Adam(copying: optimizer, to: device)
    LazyTensorBarrier()

    // Train the cell rule.
    for iteration in 0..<iterations {
      let startTime = Date()
      let steps = Int.random(in: minimumSteps...maximumSteps)
      var loggingState = initialState
      if useSamplePool {
        initialBatch = samplePool.sample(batchSize: batchSize, damaged: damagedSamples)
      }
      
      let (loss, ruleGradient) = valueWithGradient(at: cellRule) { model -> Tensor<Float> in
        var state = initialBatch
        for _ in 0..<steps {
          // Note: the following clips the X10 backward trace and is a no-op otherwise.
          state = clipBackwardsTrace(state)
          state = model(state)
          LazyTensorBarrier()
        }

        loggingState = state[0]
        if useSamplePool {
          withoutDerivative(at: cellRule) { _ in samplePool.replace(samples: state) }
        }
        return meanSquaredError(predicted: state.colorComponents, expected: paddedImageBatch)
      }
      optimizer.update(&cellRule, along: normalizeGradient(ruleGradient))
      LazyTensorBarrier()

      let lossScalar = loss.scalarized()
      print(
        "Iteration: \(iteration), loss: \(lossScalar), log loss: \(log10(lossScalar)), time: \(Date().timeIntervalSince(startTime)) s")

      if (iteration % 10) == 0 {
        LazyTensorBarrier()
        let filename = String(format: "iteration%03d", iteration)
        var state = initialState.expandingShape(at: 0)
        state = try recordGrowth(
          seed: state, rule: cellRule, steps: inferenceSteps, directory: "output", filename: filename)

        try saveImage(
          loggingState.colorComponents * 255.0, colorspace: .rgb, directory: "output", name: filename, format: .png
        )
      }

      if ((iteration + 1) % 2000) == 0 {
        optimizer.learningRate = optimizer.learningRate * 0.1
      }
    }
    
    // Perform growth using the trained model and record the results.
    var state = initialState.expandingShape(at: 0)
    state = try recordGrowth(
      seed: state, rule: cellRule, steps: inferenceSteps, directory: "output", filename: "growth")
    
    // Perform regeneration using the trained model and record the results.
    state = state.damageRightSide()
    _ = try recordGrowth(
      seed: state, rule: cellRule, steps: inferenceSteps, directory: "output", filename: "regen")
  }
}

GrowingNeuralCellularAutomata.main()
