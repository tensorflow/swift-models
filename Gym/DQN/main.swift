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

extension _Raw {
    /// Derivative of `_Raw.gatherNd`.
    ///
    /// Ported from TensorFlow Python reference implementation:
    /// https://github.com/tensorflow/tensorflow/blob/r2.2/tensorflow/python/ops/array_grad.py#L691-L701
    @inlinable
    @derivative(of: gatherNd)
    public static func vjpGatherNd<
        Scalar: TensorFlowFloatingPoint,
        Index: TensorFlowIndex
    >(
        params: Tensor<Scalar>,
        indices: Tensor<Index>
    ) -> (
        value: Tensor<Scalar>,
        pullback: (Tensor<Scalar>) -> Tensor<Scalar>
    ) {
        let shapeTensor = Tensor<Index>(params.shapeTensor)
        let value = gatherNd(params: params, indices: indices)
        return (value, { v in
            let dparams = scatterNd(indices: indices, updates: v, shape: shapeTensor)
            return dparams
        })
    }
}

// Initialize Python. This comment is a hook for internal use, do not remove.

let np = Python.import("numpy")
let gym = Python.import("gym")
let plt = Python.import("matplotlib.pyplot")

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
        actions = Tensor<Int32>(numpy: np.zeros([capacity], dtype: np.int32))!
        rewards = Tensor<Float>(numpy: np.zeros([capacity], dtype: np.float32))!
        nextStates = Tensor<Float>(numpy: np.zeros([capacity, 4], dtype: np.float32))!
        isDones = Tensor<Bool>(numpy: np.zeros([capacity], dtype: np.bool))!
    }

    func append(
        state: Tensor<Float>,
        action: Tensor<Int32>,
        reward: Tensor<Float>,
        nextState: Tensor<Float>,
        isDone: Tensor<Bool>
    ) {
        if count < capacity {
            count += 1
        }
        // Erase oldest SARS if the replay buffer is full
        states[index] = state
        actions[index] = action
        rewards[index] = reward
        nextStates[index] = nextState
        isDones[index] = isDone
        index = (index + 1) % capacity
    }

    func sample(batchSize: Int) -> (
        stateBatch: Tensor<Float>,
        actionBatch: Tensor<Int32>,
        rewardBatch: Tensor<Float>,
        nextStateBatch: Tensor<Float>,
        isDoneBatch: Tensor<Bool>
    ) {
        let randomIndices = Tensor<Int32>(numpy: np.random.randint(count, size: batchSize, dtype: np.int32))!

        let stateBatch = states.gathering(atIndices: randomIndices, alongAxis: 0)
        let actionBatch = actions.gathering(atIndices: randomIndices, alongAxis: 0)
        let rewardBatch = rewards.gathering(atIndices: randomIndices, alongAxis: 0)
        let nextStateBatch = nextStates.gathering(atIndices: randomIndices, alongAxis: 0)
        let isDoneBatch = isDones.gathering(atIndices: randomIndices, alongAxis: 0)

        return (stateBatch, actionBatch, rewardBatch, nextStateBatch, isDoneBatch)
    }
}

struct Net: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var l1, l2: Dense<Float>

    init(observationSize: Int, hiddenSize: Int, actionCount: Int) {
        l1 = Dense<Float>(inputSize: observationSize, outputSize: hiddenSize, activation: relu, weightInitializer: heNormal())
        l2 = Dense<Float>(inputSize: hiddenSize, outputSize: actionCount, activation: identity, weightInitializer: heNormal())
    }

    @differentiable
    func callAsFunction(_ input: Input) -> Output {
        return input.sequenced(through: l1, l2)
    }
}

class Agent {
    var qNet: Net
    var targetQNet: Net
    let optimizer: Adam<Net>
    let replayBuffer: ReplayBuffer
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
            let npState = np.random.randint(0, 2, dtype: np.int32)
            return Tensor<Int32>(numpy: np.array(npState, dtype: np.int32))!
        }
        else {
            // Neural network input needs to be 2D
            let tfState = Tensor<Float>(numpy: np.expand_dims(state.makeNumpyArray(), axis: 0))!
            let qValues = qNet(tfState)[0]
            return qValues[1].scalarized() > qValues[0].scalarized() ? Tensor<Int32>(1) : Tensor<Int32>(0)
        }
    }

    func train(batchSize: Int) -> Float {
        // Don't train if replay buffer is too small
        if replayBuffer.count >= minBufferSize {
            let (tfStateBatch, tfActionBatch, tfRewardBatch, tfNextStateBatch, tfIsDoneBatch) = replayBuffer.sample(batchSize: batchSize)

            let (loss, gradients) = valueWithGradient(at: qNet) { qNet -> Tensor<Float> in
                // Compute prediction batch
                let npActionBatch = tfActionBatch.makeNumpyArray()
                let npFullIndices = np.stack([np.arange(batchSize, dtype: np.int32), npActionBatch], axis: 1)
                let tfFullIndices = Tensor<Int32>(numpy: npFullIndices)!
                let stateQValueBatch = qNet(tfStateBatch)
                let predictionBatch = _Raw.gatherNd(params: stateQValueBatch, indices: tfFullIndices)

                // Compute target batch
                let targetBatch: Tensor<Float> = tfRewardBatch + Tensor<Float>(tfIsDoneBatch) * self.discount * self.targetQNet(tfNextStateBatch).max(squeezingAxes: Tensor<Int32>(1))
                return huberLoss(
                    predicted: predictionBatch,
                    expected: withoutDerivative(at: targetBatch),
                    delta: 1
                )
            }
            optimizer.update(&qNet, along: gradients)

            return loss.scalarized()
        }
        return 0
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

    init(_ env: PythonObject) {
        self.originalEnv = env
    }

    func reset() -> Tensor<Float> {
        let state = self.originalEnv.reset()
        return Tensor<Float>(numpy: np.array(state, dtype: np.float32))!
    }

    func step(_ action: Tensor<Int32>) -> (state: Tensor<Float>, reward: Tensor<Float>, isDone: Tensor<Bool>, info: PythonObject) {
        let (state, reward, isDone, info) = originalEnv.step(action.scalarized()).tuple4
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
let maxEpisode: Int = 1000
let replayBufferCapacity: Int = 5000
let minBufferSize: Int = 1000
let batchSize: Int = 64
let targetNetUpdateRate: Int = 32

// Initialize environment
let env = TensorFlowEnvironmentWrapper(gym.make("CartPole-v0"))

// Initialize agent
var qNet = Net(observationSize: 4, hiddenSize: hiddenSize, actionCount: 2)
var targetQNet = Net(observationSize: 4, hiddenSize: hiddenSize, actionCount: 2)
updateTargetQNet(source: qNet, target: &targetQNet)
let optimizer = Adam(for: qNet, learningRate: learningRate)
var replayBuffer: ReplayBuffer = ReplayBuffer(capacity: replayBufferCapacity)
var agent = Agent(qNet: qNet, targetQNet: targetQNet, optimizer: optimizer, replayBuffer: replayBuffer, discount: discount)

// RL Loop
var stepIndex = 0
var episodeIndex = 0
var episodeReturn: Float = 0
var episodeReturns: Array<Float> = []
var losses: Array<Float> = []
var state = env.reset()
while episodeIndex < maxEpisode {
    stepIndex += 1

    // Interact with environment
    let epsilon = startEpsilon * Float(maxEpisode - episodeIndex) / Float(maxEpisode)
    let action = agent.getAction(state: state, epsilon: epsilon)
    let (nextState, reward, isDone, _) = env.step(action)
    episodeReturn += reward.scalarized()

    // Save interaction to replay buffer
    replayBuffer.append(state: state, action: action, reward: reward, nextState: nextState, isDone: isDone)

    // Train agent
    losses.append(agent.train(batchSize: batchSize))

    // Periodically update Target Net
    if stepIndex % targetNetUpdateRate == 0 {
        updateTargetQNet(source: qNet, target: &targetQNet)
    }

    // End-of-episode
    if isDone.scalarized() == true {
        state = env.reset()
        episodeIndex += 1
        print(String(format: "Episode: %4d | Epsilon: %.03f | Return: %3d", episodeIndex, epsilon, Int(episodeReturn)))
        if episodeReturn > 199 {
            print("Solved in \(episodeIndex) episodes with \(stepIndex) steps!")
            break
        }
        episodeReturns.append(episodeReturn)
        episodeReturn = 0
    }

    // End-of-step
    state = nextState
}

// Save smoothed learning curve
let runningMeanWindow: Int = 2
let smoothedEpisodeReturns = np.convolve(episodeReturns, np.ones((runningMeanWindow)) / np.array(runningMeanWindow, dtype: np.int32), mode: "same")

plt.plot(smoothedEpisodeReturns)
plt.title("Deep Q-Network on CartPole-v0")
plt.xlabel("Episode")
plt.ylabel("Smoothed Episode Return")
plt.savefig("dqnSmoothedEpisodeReturns.png")
plt.clf()

// Save TD loss curve
plt.plot(losses)
plt.title("Deep Q-Network on CartPole-v0")
plt.xlabel("Step")
plt.ylabel("TD Loss")
plt.savefig("dqnTDLoss.png")
plt.clf()
