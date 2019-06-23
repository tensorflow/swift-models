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

import Python
import TensorFlow

let gym = Python.import("gym")
let environment = gym.make("Blackjack-v0")

let totalIterations = 10000
let learningPhase = totalIterations * 5 / 100

class BlackjackState {
    var playerSum: Int = 0
    var dealerCard: Int = 0
    var useableAce: Int = 0

    init(pythonState: PythonObject) {
        self.playerSum = Int(pythonState[0]) ?? 0
        self.dealerCard = Int(pythonState[1]) ?? 0
        self.useableAce = Int(pythonState[2]) ?? 0
    }
}

enum SolverType: CaseIterable {
    case random, markov, qlearning, normal
}

class Solver {
    var Q: [[[[Float]]]] = []
    var alpha: Float = 0.5
    let gamma: Float = 0.2

    let numberOfPlayerStates = 32 // 21 + 10 + 1 offset
    let numberOfDealerVisibleStates = 11 // 10 + 1 offset
    let numberOfAceStates = 2 // useable / not bool
    let numberOfPlayerActions = 2 // hit / stay

    init() {
        Q = Array(repeating: Array(repeating: Array(repeating: Array(repeating: 0.0,
                                                                     count: numberOfPlayerActions),
                                                    count: numberOfAceStates),
                                   count: numberOfDealerVisibleStates),
                  count: numberOfPlayerStates)
    }

    func updateQLearningStrategy(prior: BlackjackState,
                                 action: Int,
                                 reward: Int,
                                 post: BlackjackState) {
        let oldQ = Q[prior.playerSum][prior.dealerCard][prior.useableAce][action]
        let priorQ = (1 - alpha) * oldQ

        let maxReward = max(Q[post.playerSum][post.dealerCard][post.useableAce][0],
                            Q[post.playerSum][post.dealerCard][post.useableAce][1])
        let postQ = alpha * (Float(reward) + gamma * maxReward)

        Q[prior.playerSum][prior.dealerCard][prior.useableAce][action] += priorQ + postQ
    }

    func getQLearningStrategy(observation: BlackjackState, iteration: Int) -> Bool {
        let hitReward = Q[observation.playerSum][observation.dealerCard][observation.useableAce][0]
        let stayReward = Q[observation.playerSum][observation.dealerCard][observation.useableAce][1]

        if (iteration < Int.random(in: 1...learningPhase)) {
            return getRandomStrategy()
        } else {
            // quit learning after initial phase
            if (iteration > learningPhase) { alpha = 0.0 }
        }

        if hitReward == stayReward {
            return getRandomStrategy()
        } else {
            return hitReward < stayReward
        }
    }

    func getRandomStrategy() -> Bool {
        return Bool.random()
    }

    func getMarkovStrategy(observation: BlackjackState) -> Bool {
        // hit @ 80% probability unless over 18, in which case do the reverse
        let flip = Float.random(in: 0..<1)
        let threshHold: Float = 0.8

        if (observation.playerSum < 18) {
            return flip < threshHold
        } else {
            return flip > threshHold
        }
    }

    func getNormalStrategyLookup(playerSum: Int) -> String {
        // see figure 11: https://ieeexplore.ieee.org/document/1299399/
        switch playerSum {
        case 10: return "HHHHHSSHHH"
        case 11: return "HHSSSSSSHH"
        case 12: return "HSHHHHHHHH"
        case 13: return "HSSHHHHHHH"
        case 14: return "HSHHHHHHHH"
        case 15: return "HSSHHHHHHH"
        case 16: return "HSSSSSHHHH"
        case 17: return "HSSSSHHHHH"
        case 18: return "SSSSSSSSSS"
        case 19: return "SSSSSSSSSS"
        case 20: return "SSSSSSSSSS"
        case 21: return "SSSSSSSSSS"
        default: return "HHHHHHHHHH"
        }
    }

    func getNormalStrategy(observation: BlackjackState) -> Bool {
        if observation.playerSum == 0 {
            return true
        }
        let lookupString = getNormalStrategyLookup(playerSum: observation.playerSum)
        return Array(lookupString)[observation.dealerCard - 1] == "H"
    }

    func getStrategy(observation: BlackjackState, solver: SolverType, iteration: Int) -> Bool {
        switch solver {
        case .random:
            return getRandomStrategy()
        case .markov:
            return getMarkovStrategy(observation: observation)
        case .qlearning:
            return getQLearningStrategy(observation: observation, iteration: iteration)
        case .normal:
            return getNormalStrategy(observation: observation)
        }
    }
}

let learner = Solver()

for solver in SolverType.allCases {
    var totalReward = 0

    for i in 1...totalIterations {
        var isDone = false
        environment.reset()

        while !isDone {

            let priorState = BlackjackState(pythonState: environment._get_obs())
            let action: Int = learner.getStrategy(observation: priorState,
                                                  solver: solver,
                                                  iteration: i) ? 1 : 0

            let (pythonPostState, reward, done, _) = environment.step(action).tuple4
            let postState = BlackjackState(pythonState: pythonPostState)

            if solver == .qlearning {
                learner.updateQLearningStrategy(prior: priorState,
                                                action: action,
                                                reward: Int(reward)!,
                                                post: postState)
            }

            if done == true {
                totalReward += Int(reward) ?? 0
                isDone = true
            }
        }
    }
    print("Solver: \(solver), Total reward: \(totalReward) / \(totalIterations) trials")
}
