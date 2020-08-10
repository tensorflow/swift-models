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

func evaluate(_ agent: DeepQNetworkAgent) -> Float {
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
/// The size of the hidden layer of the 2-layer Q-network. The network has the
/// shape observationSize - hiddenSize - actionCount.
let hiddenSize: Int = 100
/// Maximum number of episodes to train the agent. The training is terminated
/// early if maximum score is achieved during evaluation.
let maxEpisode: Int = 1000
/// The initial epsilon value. With probability epsilon, the agent chooses a
/// random action instead of the action that it thinks is the best.
let epsilonStart: Float = 1
/// The terminal epsilon value.
let epsilonEnd: Float = 0.01
/// The decay rate of epsilon.
let epsilonDecay: Float = 1000
/// The learning rate for the Q-network.
let learningRate: Float = 0.001
/// The discount factor. This measures how much to "discount" the future rewards
/// that the agent will receive. The discount factor must be from 0 to 1
/// (inclusive). Discount factor of 0 means that the agent only considers the
/// immediate reward and disregards all future rewards. Discount factor of 1
/// means that the agent values all rewards equally, no matter how distant
/// in the future they may be.
let discount: Float = 0.99
/// If enabled, uses the Double DQN update equation instead of the original DQN
/// equation. This mitigates the overestimation problem of DQN. For more
/// information about Double DQN, check Deep Reinforcement Learning with Double
/// Q-learning (Hasselt, Guez, and Silver, 2015).
let useDoubleDQN: Bool = true
/// The maximum size of the replay buffer. If the replay buffer is full, the new
/// element replaces the oldest element.
let replayBufferCapacity: Int = 100000
/// The minimum replay buffer size before the training starts. Must be at least
/// the training batch size.
let minBufferSize: Int = 64
/// The training batch size.
let batchSize: Int = 64
/// If enabled, uses Combined Experience Replay (CER) sampling instead of the
/// uniform random sampling in the original DQN paper. Original DQN samples
/// batch uniformly randomly in the replay buffer. CER always includes the most
/// recent element and samples the rest of the batch uniformly randomly. This
/// makes the agent more robust to different replay buffer capacities. For more
/// information about Combined Experience Replay, check A Deeper Look at
/// Experience Replay (Zhang and Sutton, 2017).
let useCombinedExperienceReplay: Bool = true
/// The number of steps between target network updates. The target network is
/// a copy of the Q-network that is updated less frequently to stabilize the
/// training process.
let targetNetUpdateRate: Int = 5
/// The update rate for target network. In the original DQN paper, the target
/// network is updated to be the same as the Q-network. Soft target network
/// only updates the target network slightly towards the direction of the
/// Q-network. The softTargetUpdateRate of 0 means that the target network is
/// not updated at all, and 1 means that soft target network update is disabled.
let softTargetUpdateRate: Float = 0.05

// Setup device
let device: Device = Device.default

// Initialize environment
let env = TensorFlowEnvironmentWrapper(gym.make("CartPole-v0"))

// Initialize agent
var qNet = DeepQNetwork(observationSize: 4, hiddenSize: hiddenSize, actionCount: 2)
var targetQNet = DeepQNetwork(observationSize: 4, hiddenSize: hiddenSize, actionCount: 2)
let optimizer = Adam(for: qNet, learningRate: learningRate)
var replayBuffer = ReplayBuffer(
  capacity: replayBufferCapacity,
  combined: useCombinedExperienceReplay
)
var agent = DeepQNetworkAgent(
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
    let evalEpisodeReturn = evaluate(agent)
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
plt.savefig("/tmp/dqnEpisodeReturns.png")
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
plt.savefig("/tmp/dqnSmoothedEpisodeReturns.png")
plt.clf()

// // Save TD loss curve
plt.plot(losses)
plt.title("Deep Q-Network on CartPole-v0")
plt.xlabel("Step")
plt.ylabel("TD Loss")
plt.savefig("/tmp/dqnTDLoss.png")
plt.clf()
