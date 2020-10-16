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

import ModelSupport
import TensorFlow

// MARK: Visualization of shallow water equation solution

/// Visualization of the solution at a particular time-step.
struct SolutionVisualization<Solution: ShallowWaterEquationSolution> {
  let solution: Solution

  /// Returns a top-down mosaic of the water level colored by its height.
  var waterLevel: Image {
    let square = TensorShape([solution.waterLevel.count, solution.waterLevel.count, 1])
    let waterLevel = Tensor(shape: square, scalars: solution.waterLevel.flatMap { $0 })
    let normalizedWaterLevel = waterLevel.normalized(min: -1, max: +1)
    return Image(normalizedWaterLevel)
  }
}

extension ShallowWaterEquationSolution {
  var visualization: SolutionVisualization<Self> { SolutionVisualization(solution: self) }
}

// MARK: - Utilities

extension Tensor where Scalar == Float {
  /// Returns image normalized from `min`-`max` range to standard 0-255 range and converted to `UInt8`.
  fileprivate func normalized(min: Scalar = -1, max: Scalar = +1) -> Tensor<UInt8> {
    precondition(max > min)

    let clipped = self.clipped(min: min, max: max)
    let normalized = (clipped - min) / (max - min) * Float(UInt8.max)
    return Tensor<UInt8>(normalized)
  }
}
