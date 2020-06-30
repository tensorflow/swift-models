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

#if canImport(PythonKit)
    import PythonKit
#else
    import Python
#endif
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

typealias State = Tensor<Float>
typealias Action = Tensor<Int32>
typealias Reward = Tensor<Float>

class ReplayBuffer {
    var states: Tensor<Float>
    var actions: Tensor<Int32>
    var rewards: Tensor<Float>
    var nextStates: Tensor<Float>
    var isDones: Tensor<Bool>
    let capacity: Int
    var count: Int = 0
    var index: Int = 0

    init(capacity: Int) {
        self.capacity = capacity

        states = Tensor<Float>(numpy: np.zeros([capacity, 4], dtype: np.float32))!
        actions = Tensor<Int32>(numpy: np.zeros([capacity, 1], dtype: np.int32))!
        rewards = Tensor<Float>(numpy: np.zeros([capacity, 1], dtype: np.float32))!
        nextStates = Tensor<Float>(numpy: np.zeros([capacity, 4], dtype: np.float32))!
        isDones = Tensor<Bool>(numpy: np.zeros([capacity], dtype: np.bool))!
    }

    func append(state: Tensor<Float>, action: Tensor<Int32>, reward: Tensor<Float>, nextState: Tensor<Float>, isDone: Tensor<Bool>) {
        if count < capacity {
            count += 1
        }
        // Erase oldest SARS if the replay buffer is full
        states[index] = state
        actions[index] = Tensor<Int32>(numpy: np.expand_dims(action.makeNumpyArray(), axis: 0))!
        rewards[index] = Tensor<Float>(numpy: np.expand_dims(reward.makeNumpyArray(), axis: 0))!
        nextStates[index] = nextState
        isDones[index] = isDone
        index = (index + 1) % capacity
    }

    func sample(batchSize: Int) -> (stateBatch: Tensor<Float>, actionBatch: Tensor<Int32>, rewardBatch: Tensor<Float>, nextStateBatch: Tensor<Float>, isDoneBatch: Tensor<Bool>) {
        let randomIndices = Tensor<Int32>(numpy: np.random.randint(count, size: batchSize, dtype: np.int32))!

        let stateBatch = _Raw.gather(params: states, indices: randomIndices)
        let actionBatch = _Raw.gather(params: actions, indices: randomIndices)
        let rewardBatch = _Raw.gather(params: rewards, indices: randomIndices)
        let nextStateBatch = _Raw.gather(params: nextStates, indices: randomIndices)
        let isDoneBatch = _Raw.gather(params: isDones, indices: randomIndices)

        return (stateBatch, actionBatch, rewardBatch, nextStateBatch, isDoneBatch)
    }
}

struct Net: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var l1, l2: Dense<Float>

    init(observationSize: Int, hiddenSize: Int, actionCount: Int) {
        l1 = Dense<Float>(inputSize: observationSize, outputSize: hiddenSize, activation: relu, weightInitializer: heNormal())
        l2 = Dense<Float>(inputSize: hiddenSize, outputSize: actionCount, weightInitializer: heNormal())
    }

    @differentiable
    func callAsFunction(_ input: Input) -> Output {
        return input.sequenced(through: l1, l2)
    }
}

class Agent {
    // Q-network
    var qNet: Net
    // Target Q-network
    var targetQNet: Net
    // Optimizer
    let optimizer: Adam<Net>
    // Replay Buffer
    let replayBuffer: ReplayBuffer
    // Discount Factor
    let discount: Float

    init(qNet: Net, targetQNet: Net, optimizer: Adam<Net>, replayBuffer: ReplayBuffer, discount: Float) {
        self.qNet = qNet
        self.targetQNet = targetQNet
        self.optimizer = optimizer
        self.replayBuffer = replayBuffer
        self.discount = discount
    }

    func getAction(state: Tensor<Float>, epsilon: Float) -> Tensor<Int32> {
        if Float(np.random.uniform()).unwrapped() < epsilon {
            // print("getAction | state: \(state)")
            // print("getAction | epsilon: \(epsilon)")
            let npState = np.random.randint(0, 2, dtype: np.int32)
            // print("getAction | npState: \(npState)")
            return Tensor<Int32>(numpy: np.array(npState, dtype: np.int32))!
        }
        else {
            // Neural network input needs to be 2D
            let tfState = Tensor<Float>(numpy: np.expand_dims(state.makeNumpyArray(), axis: 0))!
            let qValues = qNet(tfState)
            let leftQValue = Float(qValues[0][0]).unwrapped()
            let rightQValue = Float(qValues[0][1]).unwrapped()
            return leftQValue < rightQValue ? Tensor<Int32>(numpy: np.array(1, dtype: np.int32))! : Tensor<Int32>(numpy: np.array(0, dtype: np.int32))!
        }
    }

    func train(batchSize: Int) {
        // Don't train if replay buffer is too small
        if replayBuffer.count >= batchSize {
            // print("train | Start training")
            let (tfStateBatch, tfActionBatch, tfRewardBatch, tfNextStateBatch, tfIsDoneBatch) = replayBuffer.sample(batchSize: batchSize)

            // Gradient are accumulated since we calculate every element in the batch individually
            var totalGrad = qNet.zeroTangentVector
            for i in 0..<batchSize {
                let 𝛁qNet = gradient(at: qNet) { qNet -> Tensor<Float> in

                    let stateQValueBatch = qNet(tfStateBatch)
                    let tfAction: Tensor<Int32> = tfActionBatch[i][0]
                    let action = Int(tfAction.scalarized())
                    let prediction: Tensor<Float> = stateQValueBatch[i][action]

                    let nextStateQValueBatch = self.targetQNet(tfNextStateBatch)
                    let tfReward: Tensor<Float> = tfRewardBatch[i][0]
                    let leftQValue = Float(nextStateQValueBatch[i][0].scalarized())
                    let rightQValue = Float(nextStateQValueBatch[i][1].scalarized())
                    let maxNextStateQValue = leftQValue > rightQValue ? leftQValue : rightQValue
                    let target: Tensor<Float> = tfReward + Tensor<Float>(tfIsDoneBatch[i]) * self.discount * maxNextStateQValue

                    return squaredDifference(prediction, withoutDerivative(at: target))
                }
                totalGrad += 𝛁qNet
            }
            optimizer.update(&qNet, along: totalGrad)

            // TODO: Use parallelized methods commented out below
            // TODO: _Raw.gatherNd() is not differentiable?
            // let 𝛁qNet = gradient(at: qNet) { qNet -> Tensor<Float> in
            //     // Compute prediction batch
            //     let npActionBatch = tfActionBatch.makeNumpyArray()
            //     print("A: \(np.arange(batchSize, dtype: np.int32)))")
            //     print("B: \(npActionBatch.flatten())")
            //     let npFullIndices = np.stack([np.arange(batchSize, dtype: np.int32), npActionBatch.flatten()], axis: 1)
            //     let tfFullIndices = Tensor<Int32>(numpy: npFullIndices)!
            //     let stateQValueBatch = qNet(tfStateBatch)
            //     let predictionBatch = _Raw.gatherNd(params: stateQValueBatch, indices: tfFullIndices)

            //     // TODO: Just save rewards as 1D to avoid this extra squeeze operation
            //     // Compute target batch
            //     let targetBatch: Tensor<Float> = _Raw.squeeze(tfRewardBatch, squeezeDims: [1]) + self.discount * _Raw.max(self.targetQNet(tfNextStateBatch), reductionIndices: Tensor<Int32>(1))

            //     return squaredDifference(predictionBatch, withoutDerivative(at: targetBatch))
            // }
            // optimizer.update(&qNet, along: 𝛁qNet)
        }
    }
}

func updateTargetQNet(source: Net, target: inout Net) {
    target.l1.weight = Tensor<Float>(source.l1.weight)
    target.l1.bias = Tensor<Float>(source.l1.bias)
    target.l2.weight = Tensor<Float>(source.l2.weight)
    target.l2.bias = Tensor<Float>(source.l2.bias)
}

class TensorFlowEnvironmentWrapper {
    let originalEnv: PythonObject
    let action_space: PythonObject
    let observation_space: PythonObject

    init(_ env: PythonObject) {
        self.originalEnv = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
    }

    func reset() -> Tensor<Float> {
        let state = self.originalEnv.reset()
        return Tensor<Float>(numpy: np.array(state, dtype: np.float32))!
    }

    func step(_ action: Tensor<Int32>) -> (state: Tensor<Float>, reward: Tensor<Float>, isDone: Tensor<Bool>, info: PythonObject) {
        let npAction = action.makeNumpyArray().item()
        let (state, reward, isDone, info) = originalEnv.step(npAction).tuple4
        let tfState = Tensor<Float>(numpy: np.array(state, dtype: np.float32))!
        let tfReward = Tensor<Float>(numpy: np.array(reward, dtype: np.float32))!
        let tfIsDone = Tensor<Bool>(numpy: np.array(isDone, dtype: np.bool))!
        return (tfState, tfReward, tfIsDone, info)
    }
}

// Hyperparameters
let discount: Float = 0.99
let learningRate: Float = 0.01
let hiddenSize: Int = 64
let startEpsilon: Float = 0.5
let maxEpisode: Int = 100
let replayBufferCapacity: Int = 1000
let batchSize: Int = 32
let targetNetUpdateRate: Int = 1

// Initialize environment
let env = TensorFlowEnvironmentWrapper(gym.make("CartPole-v0"))

// Initialize agent
let actionCount = Int(env.action_space.n).unwrapped()
var qNet = Net(observationSize: 4, hiddenSize: hiddenSize, actionCount: actionCount)
var targetQNet = Net(observationSize: 4, hiddenSize: hiddenSize, actionCount: actionCount)
updateTargetQNet(source: qNet, target: &targetQNet)
let optimizer = Adam(for: qNet, learningRate: learningRate)
var replayBuffer: ReplayBuffer = ReplayBuffer(capacity: replayBufferCapacity)
var agent = Agent(qNet: qNet, targetQNet: targetQNet, optimizer: optimizer, replayBuffer: replayBuffer, discount: discount)

// RL Loop
var stepIndex = 0
var episodeIndex = 0
var episodeReturn: Int = 0
var episodeReturns: Array<Int> = []
var state = env.reset()
while episodeIndex < maxEpisode {
    stepIndex += 1
    // print("Step \(stepIndex)")

    // Interact with environment
    let action = agent.getAction(state: state, epsilon: startEpsilon * Float(maxEpisode - episodeIndex))
    // print("action: \(action)")
    var (nextState, reward, isDone, _) = env.step(action)
    // print("state: \(state)")
    // print("nextState: \(nextState)")
    // print("reward: \(reward)")
    // print("isDone: \(isDone)")
    episodeReturn += Int(reward.scalarized())
    // print("episodeReturn: \(episodeReturn)")

    // Save interaction to replay buffer
    replayBuffer.append(state: state, action: action, reward: reward, nextState: nextState, isDone: isDone)
    // print("Append successful")

    // Train agent
    agent.train(batchSize: batchSize)
    // print("Train successful")

    // Periodically update Target Net
    if stepIndex % targetNetUpdateRate == 0 {
        updateTargetQNet(source: qNet, target: &targetQNet)
    }
    // print("Target net update successful")

    // End-of-episode
    if isDone.scalarized() == true {
        state = env.reset()
        episodeIndex += 1
        print("Episode \(episodeIndex) Return \(episodeReturn)")
        if episodeReturn > 199 {
            print("Solved in \(episodeIndex) episodes with \(stepIndex) steps!")
            break
        }
        episodeReturns.append(episodeReturn)
        episodeReturn = 0
    }

    // End-of-step
    nextState = state
}

// Save smoothed learning curve
let runningMeanWindow: Int = 2
let smoothedEpisodeReturns = np.convolve(episodeReturns, np.ones((runningMeanWindow)) / np.array(runningMeanWindow, dtype: np.int32), mode: "same")

plt.plot(smoothedEpisodeReturns)
plt.title("Deep Q-Network on CartPole-v0")
plt.xlabel("Episode")
plt.ylabel("Smoothed Episode Return")
plt.savefig("dqnSmoothedEpisodeReturns.png")
