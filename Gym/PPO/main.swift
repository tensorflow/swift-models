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

import PythonKit
import TensorFlow

// Force unwrapping with `!` does not provide source location when unwrapping `nil`, so we instead
// make a utility function for debuggability.
fileprivate extension Optional {
    func unwrapped(file: StaticString = #filePath, line: UInt = #line) -> Wrapped {
        guard let unwrapped = self else {
            fatalError("Value is nil", file: (file), line: line)
        }
        return unwrapped
    }
}

// Initialize Python. This comment is a hook for internal use, do not remove.

let np = Python.import("numpy")
let gym = Python.import("gym")
let plt = Python.import("matplotlib.pyplot")


let env = gym.make("CartPole-v0")
let observationSize: Int = Int(env.observation_space.shape[0])!
let actionCount: Int = Int(env.action_space.n)!

// Hyperparameters
// Network HP
let hiddenSize: Int = 128
// Optimizer HP
let learningRate: Float = 0.0003
// TODO(seungjaeryanlee): Not used
let betas: [Float] = [0.9, 0.999]
let gamma: Float = 0.99
let epochs: Int = 10
let clipEpsilon: Float = 0.1
let entropyCoefficient: Float = 0.0001
// Interaction
let maxEpisodes: Int = 1000
let maxTimesteps: Int = 200
let updateTimestep: Int = 1000
// Log
let logInterval: Int = 20
let solvedReward: Float = 199

var memory: PPOMemory = PPOMemory()
var agent: PPOAgent = PPOAgent(
    observationSize: observationSize,
    hiddenSize: hiddenSize,
    actionCount: actionCount,
    learningRate: learningRate,
    betas: betas,
    gamma: gamma,
    epochs: epochs,
    clipEpsilon: clipEpsilon,
    entropyCoefficient: entropyCoefficient
)

// Training loop
var timestep: Int = 0
var episodeReturn: Float = 0
var episodeReturns: [Float] = []
var maxEpisodeReturn: Float = -1
for episodeIndex in 1..<maxEpisodes+1 {
    var state = env.reset()
    for _ in 0..<maxTimesteps {
        timestep += 1
        let tfState: Tensor<Float> = Tensor<Float>(numpy: np.array(state, dtype: np.float32))!
        let dist: Categorical<Int32> = agent.oldActorCritic(tfState)
        let action: Int32 = dist.sample().scalarized()
        let logProb: Float = Float(dist.logProbabilities.makeNumpyArray()[action])!
        let (newState, reward, done, _) = env.step(action).tuple4

        let convertedStates: [Float] = Array(numpy: tfState.makeNumpyArray().flatten())!
        memory.states.append(convertedStates)
        memory.actions.append(action)
        memory.logProbs.append(logProb)
        memory.rewards.append(Float(reward)!)
        memory.isDones.append(Bool(done)!)

        if timestep % updateTimestep == 0 {
            agent.update(memory: memory)
            memory.clear_memory()
            timestep = 0
        }

        episodeReturn += Float(reward)!
        if Bool(done)! == true {
            episodeReturns.append(episodeReturn)
            if maxEpisodeReturn < episodeReturn {
                maxEpisodeReturn = episodeReturn
                print(String(format: "Episode: %4d | Return: %6.2f", episodeIndex, episodeReturn))
            }
            episodeReturn = 0
            break
        }

        state = newState
    }

    // Break when CartPole is solved for 10 consecutive episodes
    if episodeReturns.suffix(10).reduce(0, +) == 200 * 10 {
        print(String(format: "Solved in %d episodes!", episodeIndex))
        break
    }
}

// Save learning curve
plt.plot(episodeReturns)
plt.title("Proximal Policy Optimization on CartPole-v0")
plt.xlabel("Episode")
plt.ylabel("Episode Return")
// TODO: Save figure to /tmp/
plt.savefig("ppoEpisodeReturns.png")
plt.clf()
