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

/// Solves the FrozenLake RL problem via Q-learning. This model does not use a
/// neural net, and instead demonstrates Swift host-side numeric processing as
/// well as Python integration.

let discountRate: Float = 0.9
let learningRate: Float = 0.2
let testEpisodeCount = 20

typealias State = Int
typealias Action = Int

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
struct StateAction: Equatable, Hashable {
    let state: State
    let action: Action
}

// Comparison function for conforming to Equatable protocol.
func == (lhs: StateAction, rhs: StateAction) -> Bool {
    return lhs.state == rhs.state && lhs.action == rhs.action
}

class Agent {
    /// The number of actions.
    let actionCount: Int
    /// The current training environmental state that the agent is in.
    var state: State
    /// The "action value" (expected future reward value) of a pair of state and action.
    var values: [StateAction: Float] = [:]

    init(environment: PythonObject) {
        actionCount = Int(environment.action_space.n).unwrapped()
        state = State(environment.reset()).unwrapped()
    }
    
    func sampleEnvironment(
      environment: PythonObject
    ) -> (
      state: State,
      action: Int,
      reward: Float,
      newState: State
    ) {
        let action = environment.action_space.sample()
        let (newState, reward, isDone, _) = environment.step(action).tuple4

        let oldState = state
        if isDone == true {
            state = State(environment.reset()).unwrapped()
        } else {
            state = State(newState).unwrapped()
        }
        return (oldState,
                Int(action).unwrapped(),
                Float(reward).unwrapped(),
                State(newState).unwrapped())
    }

    func bestValueAndAction(state: State) -> (bestValue: Float, bestAction: Action) {
        var bestValue: Float = 0.0
        var bestAction: Action = -1  // initialize to an invalid value
        for action in (0..<actionCount) {
            let stateAction = StateAction(state: state, action: action)
            let actionValue = values[stateAction] ?? 0.0
            if action == 0 || bestValue < actionValue {
                bestValue = actionValue
                bestAction = action
            }
        }
        return (bestValue, bestAction)
    }

    func updateActionValue(state: State, action: Int, reward: Float, nextState: State) {
        let (bestValue, _) = bestValueAndAction(state: nextState)
        let newValue = reward + discountRate * bestValue
        let stateAction = StateAction(state: state, action: action)
        let oldValue = values[stateAction] ?? 0.0
        let updatedValue = oldValue * (1-learningRate) + newValue * learningRate
        values[stateAction] = updatedValue
    }

    func playEpisode(testEnvironment: PythonObject) -> Float {
        var totalReward: Float = 0.0
        var testState = State(testEnvironment.reset()).unwrapped()
        while true {
            let (_, action) = bestValueAndAction(state: testState)
            let (newState, reward, isDone, _) = testEnvironment.step(action).tuple4
            totalReward += Float(reward).unwrapped()
            if isDone == true {
                break
            }
            testState = State(newState).unwrapped()
        }
        return totalReward
    }
}

var iterationIndex = 0
var bestReward: Float = 0.0
let trainEnvironment = gym.make("FrozenLake-v0")
var agent = Agent(environment: trainEnvironment)
let testEnvironment = gym.make("FrozenLake-v0")
while true {
    if iterationIndex % 100 == 0 {
        print("Running iteration \(iterationIndex)")
    }
    iterationIndex += 1
    let (state, action, reward, nextState) = agent.sampleEnvironment(environment: trainEnvironment)
    agent.updateActionValue(state: state, action: action, reward: reward, nextState: nextState)

    var testReward: Float = 0.0
    for _ in 0..<testEpisodeCount {
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
