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

struct SamplePool {
  var samples: [Tensor<Float>]
  let initialState: Tensor<Float>
  
  init(initialState: Tensor<Float>, size: Int) {
    samples = [Tensor<Float>](repeating: initialState, count: size)
    self.initialState = initialState
  }
  
  // This rearranges the pool to place the randomly sampled batch upfront, for easy replacement later.
  mutating func sample(batchSize: Int, damaged: Int = 0) -> Tensor<Float> {
    for index in 0..<batchSize {
      let choice = Int.random(in: index..<samples.count)
      if index != choice {
        samples.swapAt(index, choice)
      }
    }
    
    // TODO: Have this sorted by loss.
    samples[0] = initialState
    if damaged > 0 {
      for damagedIndex in (batchSize - damaged - 1)..<batchSize {
        samples[damagedIndex] = samples[damagedIndex].applyCircleDamage()
      }
    }

    return Tensor(stacking: Array(samples[0..<batchSize]))
  }
  
  mutating func replace(samples: Tensor<Float>) {
    let samplesToInsert = samples.unstacked()
    self.samples.replaceSubrange(0..<samplesToInsert.count, with: samplesToInsert)
  }
}

extension Tensor where Scalar == Float {
  func applyCircleDamage() -> Tensor {
    let width = self.shape[self.rank - 2]
    let height = self.shape[self.rank - 3]
    let radius = Float.random(in: 0.1..<0.4)
    let centerX = Float.random(in: -0.5..<0.5)
    let centerY = Float.random(in: -0.5..<0.5)
    var x = Tensor<Float>(linearSpaceFrom: -1.0, to: 1.0, count: width, on: self.device)
    var y = Tensor<Float>(linearSpaceFrom: -1.0, to: 1.0, count: height, on: self.device)
    x = ((x - centerX) / radius).broadcasted(to: [height, width])
    y = ((y - centerY) / radius).expandingShape(at: 1).broadcasted(to: [height, width])
    let distanceFromCenter = (x * x + y * y).expandingShape(at: 2)
    let circleMask = distanceFromCenter.mask { $0 .> 1.0 }
    return self * circleMask
  }

  // TODO: Extend this to arbitrary rectangular sections.
  func damageRightSide() -> Tensor {
    let width = self.shape[self.rank - 2]
    let height = self.shape[self.rank - 3]
    var x = Tensor<Float>(linearSpaceFrom: -1.0, to: 1.0, count: width, on: self.device)
    x = x.broadcasted(to: [height, width]).expandingShape(at: 2)
    let rectangleMask = x.mask { $0 .< 0.0 }
    return self * rectangleMask
  }
}
