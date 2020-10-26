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
import Benchmark
import Foundation
import ModelSupport
import TensorFlow

// MARK: Command line interface

struct ShallowWaterPDE: ParsableCommand {
  static var configuration = CommandConfiguration(
    commandName: "Shallow-Water-PDE",
    abstract: "Solve shallow water PDE on a unit square.",
    discussion: "Animations of the solution are saved in the 'output' directory."
  )

  enum Task: String, EnumerableFlag {
    case splash, optimization, benchmark
  }
  @Flag(help: "Task to run.")
  var tasks: [Task] = [.splash]

  @Option(help: ArgumentHelp("Number of simulated values along X/Y directions.", valueName: "N"))
  var resolution = 256
    
  @Option(help: ArgumentHelp("Number of simulated time-steps.", valueName: "T"))
  var duration = 512
    
  @Option(help: ArgumentHelp("Image to use as an optimization target.", valueName: "image"))
  var target: String = "Examples/Shallow-Water-PDE/Images/Target.jpg"

  @Option(help: ArgumentHelp("Number of optimization iterations.", valueName: "I"))
  var iterations = 200
    
  @Option(help: ArgumentHelp("Learning rate for optimization.", valueName: "α"))
  var learningRate: Float = 500.0

  /// Runs a simple simulation in a rectangular bathtub initialized with Dirac delta function.
  public func runSplash() {
    var initialSplashLevel = Tensor<Float>(zeros: [resolution, resolution])
    initialSplashLevel[resolution / 2, resolution / 2] = Tensor(100)

    let initialSplash = TensorSliceSolution(waterLevel: initialSplashLevel)
    let splashEvolution = [TensorSliceSolution](evolve: initialSplash, for: duration)

    try! splashEvolution.saveAnimatedImage(directory: "output", name: "splash")
  }

  /// Runs an optimization through time-steps and updates the initial water height to obtain a specific wave patter at the end.
  public func runOptimization() {
    var initialWaterLevel = Tensor<Float>(zeros: [resolution, resolution])

    let targetImage = Image(contentsOf: URL(fileURLWithPath: self.target))
    var target = targetImage.resized(to: (resolution, resolution)).tensor - Float(UInt8.max) / 2
    target = target.mean(squeezingAxes: 2) / Float(UInt8.max)

    for opt in 1...iterations {

      let (loss, 𝛁initialWaterLevel) = valueWithGradient(at: initialWaterLevel) {
        (initialWaterLevel) -> Float in
        let initialSolution = TensorSliceSolution(waterLevel: initialWaterLevel)
        let evolution = [TensorSliceSolution](evolve: initialSolution, for: duration)

        let last = withoutDerivative(at: evolution.count - 1)
        let loss = evolution[last].meanSquaredError(to: target)
        return loss
      }

      print("\(opt): \(loss)")
      initialWaterLevel.move(along: 𝛁initialWaterLevel.scaled(by: -learningRate))
    }

    let initialSolution = TensorSliceSolution(waterLevel: initialWaterLevel)
    let evolution = [TensorSliceSolution](evolve: initialSolution, for: duration)

    try! evolution.saveAnimatedImage(directory: "output", name: "optimization")
  }

  private func runSplashArrayLoopBenchmark() {
    let waterLevelRow = [Float](repeating: 0.0, count: resolution)
    var initialWaterLevel = [[Float]](repeating: waterLevelRow, count: resolution)
    initialWaterLevel[resolution / 2][resolution / 2] = 100

    let initialSolution = ArrayLoopSolution(waterLevel: initialWaterLevel)
    _ = [ArrayLoopSolution](evolve: initialSolution, for: duration)
  }

  private func runSplashTensorLoopBenchmark(on device: Device) {
    var initialWaterLevel = Tensor<Float>(zeros: [resolution, resolution], on: device)
    initialWaterLevel[resolution / 2][resolution / 2] = Tensor<Float>(100, on: device)

    let initialSolution = TensorLoopSolution(waterLevel: initialWaterLevel)
    _ = [TensorLoopSolution](evolve: initialSolution, for: duration)
  }

  private func runSplashTensorSliceBenchmark(on device: Device) {
    var initialWaterLevel = Tensor<Float>(zeros: [resolution, resolution], on: device)
    initialWaterLevel[resolution / 2][resolution / 2] = Tensor<Float>(100, on: device)

    let initialSolution = TensorSliceSolution(waterLevel: initialWaterLevel)
    _ = [TensorSliceSolution](evolve: initialSolution, for: duration)
  }

  private func runSplashTensorConvBenchmark(on device: Device) {
    var initialWaterLevel = Tensor<Float>(zeros: [resolution, resolution], on: device)
    initialWaterLevel[resolution / 2][resolution / 2] = Tensor<Float>(100, on: device)

    let initialSolution = TensorConvSolution(waterLevel: initialWaterLevel)
    _ = [TensorConvSolution](evolve: initialSolution, for: duration)
  }

  /// Benchmark suite that exercises the 3 different solver implementations on a simple problem without back-propagation.
  public var splashBenchmarks: BenchmarkSuite {
    BenchmarkSuite(
      name: "Shallow Water PDE Solver",
      settings: Iterations(10), WarmupIterations(2)
    ) { suite in
      suite.benchmark("Array Loop") {
        runSplashArrayLoopBenchmark()
      }

      //            FIXME: This is at least 1000x slower. One can easily grow old while waiting... :(
      //            suite.benchmark("Tensor Loop") {
      //                runSplashTensorLoopBenchmark(on: Device.default)
      //            }
      //            suite.benchmark("Tensor Loop (XLA)") {
      //                runSplashTensorLoopBenchmark(on: Device.defaultXLA)
      //            }

      suite.benchmark("Tensor Slice") {
        runSplashTensorSliceBenchmark(on: Device.default)
      }
      suite.benchmark("Tensor Slice (XLA)") {
        runSplashTensorSliceBenchmark(on: Device.defaultXLA)
      }

      suite.benchmark("Tensor Conv") {
        runSplashTensorConvBenchmark(on: Device.default)
      }
      suite.benchmark("Tensor Conv (XLA)") {
        runSplashTensorConvBenchmark(on: Device.defaultXLA)
      }
    }
  }

  mutating func run() throws {
    for task in tasks {
      switch task {
      case .splash:
        runSplash()
      case .optimization:
        runOptimization()
      case .benchmark:
        var runner = BenchmarkRunner(
          suites: [splashBenchmarks], settings: [TimeUnit(.ms)], customDefaults: [])
        try runner.run()
      }
    }
  }
}

// MARK: - Main

ShallowWaterPDE.main()
