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
  @noDerivative var waterLevel: [[Float]] { get }
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
}
