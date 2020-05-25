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
import TensorFlow

struct FractalCommand: ParsableCommand {
  static var configuration = CommandConfiguration(
    commandName: "Fractals",
    abstract: """
      Computes fractals of a variety of types and writes an image from the results.
      """,
    subcommands: [
      JuliaSubcommand.self,
      MandelbrotSubcommand.self,
    ])
}

extension FractalCommand {
  struct JuliaSubcommand: ParsableCommand {
    static var configuration = CommandConfiguration(
      commandName: "JuliaSet",
      abstract: "Calculate and save an image of the Julia set.")

    @Flag(help: "Use eager backend.")
    var eager: Bool

    @Flag(help: "Use X10 backend.")
    var x10: Bool

    @Option(help: "Number of iterations to run.")
    var iterations: Int?

    @Option(help: "The region of complex numbers to operate over.")
    var region: ComplexRegion?

    @Option(help: "Tolerance threshold to mark divergence.")
    var tolerance: Float?

    @Option(help: "Complex constant.")
    var constant: ComplexConstant?

    @Option(help: "Output image file.")
    var outputFile: String?

    @Option(help: "Output image size.")
    var imageSize: ImageSize?

    func validate() throws {
      guard !(eager && x10) else {
        throw ValidationError(
          "Can't specify both --eager and --x10 backends.")
      }
    }

    func run() throws {
      let device: Device
      if x10 {
        device = Device.defaultXLA
      } else {
        device = Device.defaultTFEager
      }

      let divergenceGrid = juliaSet(
        iterations: iterations ?? 200,
        constant: constant ?? ComplexConstant(real: -0.8, imaginary: 0.156),
        tolerance: tolerance ?? 4.0,
        region: region
          ?? ComplexRegion(
            realMinimum: -1.7, realMaximum: 1.7, imaginaryMinimum: -1.7, imaginaryMaximum: 1.7),
        imageSize: imageSize ?? ImageSize(width: 1030, height: 1030), device: device)

      do {
        try saveFractalImage(
          divergenceGrid, iterations: iterations ?? 200, fileName: outputFile ?? "julia")
      } catch {
        print("Error saving fractal image: \(error)")
      }
    }
  }
}

extension FractalCommand {
  struct MandelbrotSubcommand: ParsableCommand {
    static var configuration = CommandConfiguration(
      commandName: "MandelbrotSet",
      abstract: "Calculate and save an image of the Mandelbrot set.")

    @Flag(help: "Use eager backend.")
    var eager: Bool

    @Flag(help: "Use X10 backend.")
    var x10: Bool

    @Option(help: "Number of iterations to run.")
    var iterations: Int?

    @Option(help: "The region of complex numbers to operate over.")
    var region: ComplexRegion?

    @Option(help: "Tolerance threshold to mark divergence.")
    var tolerance: Float?

    @Option(help: "Output image file.")
    var outputFile: String?

    @Option(help: "Output image size.")
    var imageSize: ImageSize?

    func validate() throws {
      guard !(eager && x10) else {
        throw ValidationError(
          "Can't specify both --eager and --x10 backends.")
      }
    }

    func run() throws {
      let device: Device
      if x10 {
        device = Device.defaultXLA
      } else {
        device = Device.defaultTFEager
      }
      let divergenceGrid = mandelbrotSet(
        iterations: iterations ?? 200, tolerance: tolerance ?? 4.0,
        region: region
          ?? ComplexRegion(
            realMinimum: -2.0, realMaximum: 1.0, imaginaryMinimum: -1.3, imaginaryMaximum: 1.3),
        imageSize: imageSize ?? ImageSize(width: 1030, height: 1030), device: device)

      do {
        try saveFractalImage(
          divergenceGrid, iterations: iterations ?? 200, fileName: outputFile ?? "mandelbrot")
      } catch {
        print("Error saving fractal image: \(error)")
      }
    }
  }
}

FractalCommand.main()
