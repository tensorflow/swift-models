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

import Foundation
import PythonKit
import TensorFlow

// Initialize Python. This comment is a hook for internal use, do not remove.

let np = Python.import("numpy")
let gym = Python.import("gym")
let plt = Python.import("matplotlib.pyplot")

class TensorFlowEnvironmentWrapper {
  let originalEnv: PythonObject

  init(_ env: PythonObject) {
    self.originalEnv = env
  }

  func reset() -> Tensor<Float> {
    let state = self.originalEnv.reset()
    return Tensor<Float>(numpy: np.array(state, dtype: np.float32))!
  }

  func step(_ action: Tensor<Int32>) -> (
    state: Tensor<Float>, reward: Tensor<Float>, isDone: Tensor<Bool>, info: PythonObject
  ) {
    let (state, reward, isDone, info) = originalEnv.step(action.scalarized()).tuple4
    let tfState = Tensor<Float>(numpy: np.array(state, dtype: np.float32))!
    let tfReward = Tensor<Float>(numpy: np.array(reward, dtype: np.float32))!
    let tfIsDone = Tensor<Bool>(numpy: np.array(isDone, dtype: np.bool))!
    return (tfState, tfReward, tfIsDone, info)
  }
}

func eval(agent: Agent) -> Float {
  let evalEnv = TensorFlowEnvironmentWrapper(gym.make("CartPole-v0"))
  var evalEpisodeReturn: Float = 0
  var state: Tensor<Float> = evalEnv.reset()
  var reward: Tensor<Float>
  var evalIsDone: Tensor<Bool> = Tensor<Bool>(false)
  while evalIsDone.scalarized() == false {
    let action = agent.getAction(state: state, epsilon: 0)
    (state, reward, evalIsDone, _) = evalEnv.step(action)
    evalEpisodeReturn += reward.scalarized()
  }

  return evalEpisodeReturn
}

// Hyperparameters
// - Network Hyperparameters
let hiddenSize: Int = 100
// - Agent-Env Interaction Hyperparameters
let maxEpisode: Int = 1000
let epsilonStart: Float = 0.1
let epsilonEnd: Float = 0.1
let epsilonDecay: Float = 10000
// - Update Hyperparameters
let learningRate: Float = 0.001
let discount: Float = 0.99
let useDoubleDQN: Bool = true
// - Replay Buffer Hyperparameters
let replayBufferCapacity: Int = 100000
let minBufferSize: Int = 64
let batchSize: Int = 64
let useCombinedExperienceReplay: Bool = true
// - Target Network Hyperparameters
let targetNetUpdateRate: Int = 5
let softTargetUpdateRate: Float = 0.05

// Setup device
let device: Device = Device.default

// Initialize environment
let env = TensorFlowEnvironmentWrapper(gym.make("CartPole-v0"))

// Initialize agent
var qNet = Net(observationSize: 4, hiddenSize: hiddenSize, actionCount: 2)
var targetQNet = Net(observationSize: 4, hiddenSize: hiddenSize, actionCount: 2)
let optimizer = Adam(for: qNet, learningRate: learningRate)
var replayBuffer = ReplayBuffer(
  capacity: replayBufferCapacity,
  combined: useCombinedExperienceReplay
)
var agent = Agent(
  qNet: qNet,
  targetQNet: targetQNet,
  optimizer: optimizer,
  replayBuffer: replayBuffer,
  discount: discount,
  minBufferSize: minBufferSize,
  doubleDQN: useDoubleDQN,
  device: device
)

// RL Loop
var stepIndex = 0
var episodeIndex = 0
var episodeReturn: Float = 0
var episodeReturns: [Float] = []
var losses: [Float] = []
var state = env.reset()
var bestReturn: Float = 0
while episodeIndex < maxEpisode {
  stepIndex += 1

  // Interact with environment
  let epsilon: Float =
    epsilonEnd + (epsilonStart - epsilonEnd) * exp(-1.0 * Float(stepIndex) / epsilonDecay)
  let action = agent.getAction(state: state, epsilon: epsilon)
  let (nextState, reward, isDone, _) = env.step(action)
  episodeReturn += reward.scalarized()

  // Save interaction to replay buffer
  replayBuffer.append(
    state: state, action: action, reward: reward, nextState: nextState, isDone: isDone)

  // Train agent
  losses.append(agent.train(batchSize: batchSize))

  // Periodically update Target Net
  if stepIndex % targetNetUpdateRate == 0 {
    agent.updateTargetQNet(tau: softTargetUpdateRate)
  }

  // End-of-episode
  if isDone.scalarized() == true {
    state = env.reset()
    episodeIndex += 1
    let evalEpisodeReturn = eval(agent: agent)
    episodeReturns.append(evalEpisodeReturn)
    if evalEpisodeReturn > bestReturn {
      print(
        String(
          format: "Episode: %4d | Step %6d | Epsilon: %.03f | Train: %3d | Eval: %3d", episodeIndex,
          stepIndex, epsilon, Int(episodeReturn), Int(evalEpisodeReturn)))
      bestReturn = evalEpisodeReturn
    }
    if evalEpisodeReturn > 199 {
      print("Solved in \(episodeIndex) episodes with \(stepIndex) steps!")
      break
    }
    episodeReturn = 0
  }

  // End-of-step
  state = nextState
}

// Save learning curve
plt.plot(episodeReturns)
plt.title("Deep Q-Network on CartPole-v0")
plt.xlabel("Episode")
plt.ylabel("Episode Return")
plt.savefig("dqnEpisodeReturns.png")
plt.clf()

// Save smoothed learning curve
let runningMeanWindow: Int = 10
let smoothedEpisodeReturns = np.convolve(
  episodeReturns, np.ones((runningMeanWindow)) / np.array(runningMeanWindow, dtype: np.int32),
  mode: "same")

plt.plot(episodeReturns)
plt.title("Deep Q-Network on CartPole-v0")
plt.xlabel("Episode")
plt.ylabel("Smoothed Episode Return")
plt.savefig("dqnSmoothedEpisodeReturns.png")
plt.clf()

// // Save TD loss curve
plt.plot(losses)
plt.title("Deep Q-Network on CartPole-v0")
plt.xlabel("Step")
plt.ylabel("TD Loss")
plt.savefig("dqnTDLoss.png")
plt.clf()
