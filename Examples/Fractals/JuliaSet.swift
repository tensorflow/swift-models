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
import TensorFlow

struct ComplexConstant {
  let real: Float
  let imaginary: Float
}

func juliaSet(
  iterations: Int, constant: ComplexConstant, tolerance: Float, region: ComplexRegion,
  imageSize: ImageSize, device: Device
) -> Tensor<Float> {
  let xs = Tensor<Float>(
    linearSpaceFrom: region.realMinimum, to: region.realMaximum, count: imageSize.width, on: device
  ).broadcasted(to: [imageSize.width, imageSize.height])
  let ys = Tensor<Float>(
    linearSpaceFrom: region.imaginaryMaximum, to: region.imaginaryMinimum, count: imageSize.height,
    on: device
  ).expandingShape(at: 1).broadcasted(to: [imageSize.width, imageSize.height])
  var Z = ComplexTensor(real: xs, imaginary: ys)
  let C = ComplexTensor(
    real: Tensor<Float>(repeating: constant.real, shape: xs.shape, on: device),
    imaginary: Tensor<Float>(repeating: constant.imaginary, shape: xs.shape, on: device))
  var divergence = Tensor<Float>(repeating: Float(iterations), shape: xs.shape, on: device)

  // We'll make sure the initialization of these tensors doesn't carry
  // into the trace for the first iteration.
  LazyTensorBarrier()

  let start = Date()
  var firstIteration = Date()

  for iteration in 0..<iterations {
    Z = Z * Z + C

    let aboveThreshold = abs(Z) .> tolerance
    divergence = divergence.replacing(
      with: min(divergence, Float(iteration)), where: aboveThreshold)

    // We're cutting the trace to be a single iteration.
    LazyTensorBarrier()
    if iteration == 1 {
      firstIteration = Date()
    }
  }

  print(
    "Total calculation time: \(String(format: "%.3f", Date().timeIntervalSince(start))) seconds")
  print(
    "Time after first iteration: \(String(format: "%.3f", Date().timeIntervalSince(firstIteration))) seconds"
  )

  return divergence
}

extension ComplexConstant: ExpressibleByArgument {
  init?(argument: String) {
    let subArguments = argument.split(separator: ",").compactMap { Float(String($0)) }
    guard subArguments.count >= 2 else { return nil }

    self.real = subArguments[0]
    self.imaginary = subArguments[1]
  }

  var defaultValueDescription: String {
    "\(self.real) \(self.imaginary)"
  }
}
