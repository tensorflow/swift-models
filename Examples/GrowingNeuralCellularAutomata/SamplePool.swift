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
  mutating func sample(batchSize: Int) -> Tensor<Float> {
    for index in 0..<batchSize {
      let choice = Int.random(in: index..<samples.count)
      if index != choice {
        samples.swapAt(index, choice)
      }
    }
    
    // TODO: Have this sorted by loss.
    samples[0] = initialState
    return Tensor(stacking: Array(samples[0..<batchSize]))
  }
  
  mutating func replace(samples: Tensor<Float>) {
    let samplesToInsert = samples.unstacked()
    self.samples.replaceSubrange(0..<samplesToInsert.count, with: samplesToInsert)
  }
}
