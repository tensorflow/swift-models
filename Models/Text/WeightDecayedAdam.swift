// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
import x10_optimizers_optimizer

/// Adam optimizer with weight decay.
///
/// Reference: ["Adam - A Method for Stochastic Optimization"](
/// https://arxiv.org/abs/1412.6980v8)
public func makeWeightDecayedAdam(
  learningRate: Float = 0.01,
  beta1: Float = 0.9,
  beta2: Float = 0.999,
  weightDecayRate: Float = 0.01,
  epsilon: Float = 1e-6
) -> ParameterGroupOptimizer {
  var b = ParameterGroupOptimizerBuilder()
  let lr = b.makeParameter("learningRate", learningRate)
  let beta1 = b.makeParameter("beta1", beta1)
  let beta2 = b.makeParameter("beta2", beta2)
  let wd = b.makeParameter("weightDecay", weightDecayRate)

  let firstMoment = b[state: "firstMoment"]
  let secondMoment = b[state: "secondMoment"]

  b.appendCallback { (state: inout OptimizerWeightStepState, optState: inout OptimizerState) in
    optState[state, firstMoment] =
      state[beta1] * optState[state, firstMoment] + state.grad * (1 - state[beta1])
  }

  b.appendCallback { (state: inout OptimizerWeightStepState, optState: inout OptimizerState) in
    optState[state, secondMoment] =
      state[beta2] * optState[state, secondMoment] + state.grad .* state.grad * (1 - state[beta2])
  }

  b.appendCallback { (state: inout OptimizerWeightStepState, optState: inout OptimizerState) in
    let denominator = sqrt(optState[state, secondMoment]).adding(epsilon)
    let update = optState[state, firstMoment] ./ denominator + state.weight * state[wd]
    state.step = -state[lr] * update
  }

  return b.makeOptimizer()
}
