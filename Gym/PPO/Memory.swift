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

/// A cache saving all rollouts for batch updates.
///
/// PPO first collects fixed-length trajectory segments then updates weights. All the trajectory
/// segments are discarded after the update.
struct PPOMemory {
    /// The states that the agent observed.
    var states: [[Float]] = []
    /// The actions that the agent took.
    var actions: [Int32] = []
    /// The rewards that the agent received from the environment after taking
    /// an action.
    var rewards: [Float] = []
    /// The log probabilities of the chosen action.
    var logProbs: [Float] = []
    /// The episode-terminal flag that the agent received after taking an action.
    var isDones: [Bool] = []

    init() {}

    mutating func append(state: [Float], action: Int32, reward: Float, logProb: Float, isDone: Bool) {
        states.append(state)
        actions.append(action)
        logProbs.append(logProb)
        rewards.append(reward)
        isDones.append(isDone)
    }

    mutating func removeAll() {
        states.removeAll()
        actions.removeAll()
        rewards.removeAll()
        logProbs.removeAll()
        isDones.removeAll()
    }
}
