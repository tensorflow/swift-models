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
/// The size of the hidden layer of the 2-layer actor network and critic network. The actor network
/// has the shape observationSize - hiddenSize - actionCount, and the critic network has the same
/// shape but with a single output node.
let hiddenSize: Int = 128
/// The learning rate for both the actor and the critic.
let learningRate: Float = 0.0003
/// The discount factor. This measures how much to "discount" the future rewards
/// that the agent will receive. The discount factor must be from 0 to 1
/// (inclusive). Discount factor of 0 means that the agent only considers the
/// immediate reward and disregards all future rewards. Discount factor of 1
/// means that the agent values all rewards equally, no matter how distant
/// in the future they may be. Denoted gamma in the PPO paper.
let discount: Float = 0.99
/// Number of epochs to run minibatch updates once enough trajectory segments are collected. Denoted
/// K in the PPO paper.
let epochs: Int = 10
/// Parameter to clip the probability ratio. The ratio is clipped to [1-clipEpsilon, 1+clipEpsilon].
/// Denoted epsilon in the PPO paper.
let clipEpsilon: Float = 0.1
/// Coefficient for the entropy bonus added to the objective. Denoted c_2 in the PPO paper.
let entropyCoefficient: Float = 0.0001
/// Maximum number of episodes to train the agent. The training is terminated
/// early if maximum score is achieved consecutively 10 times.
let maxEpisodes: Int = 1000
/// Maximum timestep per episode.
let maxTimesteps: Int = 200
/// The length of the trajectory segment. Denoted T in the PPO paper.
let updateTimestep: Int = 1000

var memory: PPOMemory = PPOMemory()
var agent: PPOAgent = PPOAgent(
    observationSize: observationSize,
    hiddenSize: hiddenSize,
    actionCount: actionCount,
    learningRate: learningRate,
    discount: discount,
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
        let logProb: Float = dist.logProbabilities[Int(action)].scalarized()
        let (newState, reward, isDone, _) = env.step(action).tuple4

        memory.append(
            state: Array(state)!,
            action: action,
            reward: Float(reward)!,
            logProb: logProb,
            isDone: Bool(isDone)!
        )

        if timestep % updateTimestep == 0 {
            agent.update(memory: &memory)
            memory.removeAll()
            timestep = 0
        }

        episodeReturn += Float(reward)!
        if Bool(isDone)! {
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
plt.savefig("/tmp/ppoEpisodeReturns.png")
plt.clf()
