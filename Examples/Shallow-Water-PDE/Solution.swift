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

// MARK: Solution of shallow water equation

/// Differentiable solution of shallow water equation on a unit square.
protocol ShallowWaterEquationSolution: Differentiable {
  /// Snapshot of water level height at time `time`.
  @noDerivative var waterLevel: Tensor<Float> { get }
  /// Solution time
  @noDerivative var time: Float { get }

  /// Returns solution evolved forward in time by one step.
  @differentiable
  func evolved() -> Self
}

// MARK: - Evolution of the solution in time

extension Array where Array.Element: ShallowWaterEquationSolution {

  /// Creates an array of shallow water equation solutions by evolving the `initialSolution` forward `numSteps`-times.
  @differentiable
  init(evolve initialSolution: Array.Element, for numSteps: Int) {
    self.init()

    var currentSolution = initialSolution
    for _ in 0..<numSteps {
      self.append(currentSolution)
      currentSolution = currentSolution.evolved()
    }
    self.append(currentSolution)
  }

  /// Saves the shallow water equation solutions as a GIF image with the specified `delay` and keeping every `keep` frames.
  func saveAnimatedImage(directory: String, name: String, delay: Int = 4, keep: Int = 8) throws {
    let filtered = self.enumerated().filter { $0.offset % keep == 0 }.map { $0.element }
    let frames = filtered.map { $0.waterLevel.normalizedGrayscaleImage(min: -1, max: +1) }
    try frames.saveAnimatedImage(directory: directory, name: name, delay: delay)
  }
}

// MARK: - Utilities

extension Tensor where Scalar == Float {
  /// Returns a 3D grayscale image tensor clipped and normalized from `min`-`max` to 0-255  range.
  func normalizedGrayscaleImage(min: Scalar = -1, max: Scalar = +1) -> Tensor<Float> {
    precondition(max > min)
    precondition(rank == 2)

    let clipped = self.clipped(min: min, max: max)
    let normalized = (clipped - min) / (max - min) * Float(255.0)
    
    return normalized.expandingShape(at: 2)
  }
}
