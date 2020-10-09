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
    discussion: "Solve shallow water PDE on a unit square."
  )

  enum Task: String, EnumerableFlag {
    case splash, optimization, benchmark
  }
  enum CodingKeys: String, CodingKey {
    case tasks
  }
  @Flag(help: "Task to run.")
  var tasks: [Task] = [.splash]

  let n = 256
  let duration = 512

  /// Runs a simple simulation in a rectangular bathtub initialized with Dirac delta function.
  public func runSplash() {
    var initialSplashLevel = Tensor<Float>(zeros: [n, n])
    initialSplashLevel[n / 2, n / 2] = Tensor(100)

    let initialSplash = TensorSliceSolution(waterLevel: initialSplashLevel)
    let splashEvolution = [TensorSliceSolution](evolve: initialSplash, for: duration)

    for (i, solution) in splashEvolution.enumerated() {
      let file = URL(fileURLWithPath: "Images/Splash-\(String(format: "%03d", i)).jpg")
      solution.visualization.waterLevel.save(
        to: file, colorspace: .grayscale, format: .jpeg(quality: 100))
    }
  }

  /// Runs an optimization through time-steps and updates the initial water height to obtain a specific wave patter at the end.
  public func runOptimization() {
    let α: Float = 500.0
    var initialWaterLevel = Tensor<Float>(zeros: [n, n])

    let targetImage = Image(contentsOf: URL(fileURLWithPath: "Images/Target.jpg"))
    var target = targetImage.tensor - Float(UInt8.max) / 2
    target = target.mean(squeezingAxes: 2) / Float(UInt8.max)

    for opt in 1...200 {

      let (loss, 𝛁initialWaterLevel) = valueWithGradient(at: initialWaterLevel) {
        (initialWaterLevel) -> Float in
        let initialSolution = TensorSliceSolution(waterLevel: initialWaterLevel)
        let evolution = [TensorSliceSolution](evolve: initialSolution, for: duration)

        let last = withoutDerivative(at: evolution.count - 1)
        let loss = evolution[last].meanSquaredError(to: target)
        return loss
      }

      print("\(opt): \(loss)")
      initialWaterLevel.move(along: 𝛁initialWaterLevel.scaled(by: -α))
    }

    let initialSolution = TensorSliceSolution(waterLevel: initialWaterLevel)
    let evolution = [TensorSliceSolution](evolve: initialSolution, for: duration)

    for (i, solution) in evolution.enumerated() {
      let file = URL(fileURLWithPath: "Images/Optimization-\(String(format: "%03d", i)).jpg")
      solution.visualization.waterLevel.save(
        to: file, colorspace: .grayscale, format: .jpeg(quality: 100))
    }
  }

  private func runSplashArrayLoopBenchmark() {
    var initialWaterLevel = [[Float]](repeating: [Float](repeating: 0.0, count: n), count: n)
    initialWaterLevel[n / 2][n / 2] = 100

    let initialSolution = ArrayLoopSolution(waterLevel: initialWaterLevel)
    _ = [ArrayLoopSolution](evolve: initialSolution, for: duration)
  }

  private func runSplashTensorLoopBenchmark(on device: Device) {
    var initialWaterLevel = Tensor<Float>(zeros: [n, n], on: device)
    initialWaterLevel[n / 2][n / 2] = Tensor<Float>(100, on: device)

    let initialSolution = TensorLoopSolution(waterLevel: initialWaterLevel)
    _ = [TensorLoopSolution](evolve: initialSolution, for: duration)
  }

  private func runSplashTensorSliceBenchmark(on device: Device) {
    var initialWaterLevel = Tensor<Float>(zeros: [n, n], on: device)
    initialWaterLevel[n / 2][n / 2] = Tensor<Float>(100, on: device)

    let initialSolution = TensorSliceSolution(waterLevel: initialWaterLevel)
    _ = [TensorSliceSolution](evolve: initialSolution, for: duration)
  }

  private func runSplashTensorConvBenchmark(on device: Device) {
    var initialWaterLevel = Tensor<Float>(zeros: [n, n], on: device)
    initialWaterLevel[n / 2][n / 2] = Tensor<Float>(100, on: device)

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
