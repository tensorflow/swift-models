// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

import Python
import TensorFlow

let np = Python.import("numpy")
let gym = Python.import("gym")

_RuntimeConfig.printsDebugLog = false
_RuntimeConfigP.printsDebugLog = false

/// Solves the FrozenLake RL problem via Q-learning. This model does not use a
/// neural net, and instead demonstrates Swift host-side numeric processing as
/// well as python integration.

let discountRate: Float = 0.9
let learningRate: Float = 0.2
let testEpisodeCount = 20

typealias StateT = Int64
typealias ActionT = Int

// Force unwrapping with ! does not provide source location when unwrapping
// nil, so we instead make a util function for debuggability.
fileprivate extension Optional {
  func unwrapped(file: StaticString = #file, line: UInt = #line) -> Wrapped {
    guard let unwrapped = self else {
      fatalError("Value is nil", file: file, line: line)
    }
    return unwrapped
  }
}

// This struct is defined so that `StateAction` can be a dictionary key
// type. Swift tuples cannot be dictionary key types.
struct StateAction : Hashable {
  let values : (StateT, ActionT)

  var hashValue : Int {
    get {
      let (state, action) = values
      return state.hashValue &* 31 &+ action.hashValue
    }
  }
}

// Comparison function for conforming to Equatable protocol.
func ==(lhs: StateAction, rhs: StateAction) -> Bool {
  return lhs.values == rhs.values
}

class Agent {
  let env: PythonObject

  let actionCount: Int

  var state: StateT

  /// The "action value" (expected future reward value) of a (state, action)
  /// pair.
  var values: [StateAction : Float] = [:]

  init() {
    env = gym.make("FrozenLake-v0")
    actionCount = Int(env.action_space.n).unwrapped()
    state = StateT(env.reset()).unwrapped()
  }
  
  func sampleEnvironment() -> (state: StateT, action: Int, reward: Float, newState: StateT) {
    let action = env.action_space.sample()
    let (newState, reward, isDone, _) = env.step(action).tuple4

    // if reward > 0.0 {
    //   print("cur state: \(state), new state: \(newState), reward: \(reward), isDone: \(isDone)")
    // }
    let oldState = state
    if isDone == true {
      state = StateT(env.reset()).unwrapped()
    } else {
      state = StateT(newState).unwrapped()
    }
    return (oldState,
            Int(action).unwrapped(),
            Float(reward).unwrapped(),
            StateT(newState).unwrapped())
  }

  func bestValueAndAction(state: StateT) -> (bestValue: Float, bestAction: ActionT) {
    var bestValue: Float = 0.0
    var bestAction: ActionT = -1  // initialize to an invalid value
    for action in (0..<actionCount) {
      let stateAction = StateAction(values: (state, action))
      let actionValue = values[stateAction] ?? 0.0
      if action == 0 || bestValue < actionValue {
        bestValue = actionValue
        bestAction = action
      }
    }
    // if bestValue > 0.0 {
    //   print("In state \(state), bestValue: \(bestValue), bestAction: \(bestAction)")
    // }
    return (bestValue, bestAction)
  }

  func updateActionValue(state: StateT, action: Int, reward: Float, nextState: StateT) {
    let (bestValue, _) = bestValueAndAction(state: nextState)
    let newValue = reward + discountRate * bestValue
    let stateAction = StateAction(values: (state, action))
    let oldValue = values[stateAction] ?? 0.0
    let updatedValue = oldValue * (1-learningRate) + newValue * learningRate
    values[stateAction] = updatedValue
    // if updatedValue > 0.0 {
    //   print("For (state, action) (\(s), \(a)): updated action value from \(bestValue) to \(updatedValue)")
    // }
  }

  func playEpisode(testEnvironment: PythonObject) -> Float {
    var totalReward: Float = 0.0
    var testState = StateT(testEnvironment.reset()).unwrapped()
    while true {
      let (_, action) = bestValueAndAction(state: testState)
      let (newState, reward, isDone, _) = testEnvironment.step(action).tuple4
      totalReward += Float(reward).unwrapped()
      if isDone == true {
        break
      }
      testState = StateT(newState).unwrapped()
    }
    // if totalReward > 0.0 {
    //   print("Got \(totalReward) reward when playing an episode.")
    // }
    return totalReward
  }
}

var iterationIndex = 0
var bestReward: Float = 0.0
var agent = Agent()
let testEnvironment = gym.make("FrozenLake-v0")
while true {
  if iterationIndex % 100 == 0 {
    print("Running iter \(iterationIndex)")
  }
  iterationIndex += 1
  let (state, action, reward, nextState) = agent.sampleEnvironment()
  agent.updateActionValue(state: state, action: action, reward: reward, nextState: nextState)

  var testReward: Float = 0.0
  for _ in (0..<testEpisodeCount) {
    testReward += agent.playEpisode(testEnvironment: testEnvironment)
  }
  testReward /= Float(testEpisodeCount)
  if testReward > bestReward {
    print("Best reward updated \(bestReward) -> \(testReward)")
    bestReward = testReward
  }
  if testReward > 0.80 {
    print("Solved in \(iterationIndex) iterations!")
    break
  }
}
