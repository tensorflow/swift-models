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

let np = Python.import("numpy")
let gym = Python.import("gym")

/// Model parameters and hyperparameters.
let hiddenSize = 128
let batchSize = 16
/// Controls the amount of good/long episodes to retain for training.
let percentile = 70

// Force unwrapping with `!` does not provide source location when unwrapping `nil`, so we instead
// make a utility function for debuggability.
fileprivate extension Optional {
    func unwrapped(file: StaticString = #file, line: UInt = #line) -> Wrapped {
        guard let unwrapped = self else {
            fatalError("Value is nil", file: file, line: line)
        }
        return unwrapped
    }
}

/// A simple two layer dense net.
struct Net: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var l1, l2: Dense<Float>

    init(observationSize: Int, hiddenSize: Int, actionCount: Int) {
        l1 = Dense<Float>(inputSize: observationSize, outputSize: hiddenSize, activation: relu)
        l2 = Dense<Float>(inputSize: hiddenSize, outputSize: actionCount)
    }

    @differentiable
    func callAsFunction(_ input: Input) -> Output {
        return input.sequenced(through: l1, l2)
    }
}

/// An episode is a list of steps, where each step records the observation from
/// env and the action taken. They will serve respectively as the input and
/// target (label) of the neural net training.
struct Episode {
    struct Step {
        let observation: Tensor<Float>
        let action: Int32
    }

    let steps: [Step]
    let reward: Float
}

/// Filtering out bad/short episodes before we feed them as neural net training data.
func filteringBatch(
  episodes: [Episode],
  actionCount: Int
) -> (input: Tensor<Float>, target: Tensor<Float>, episodeCount: Int, meanReward: Float) {
    let rewards = episodes.map { $0.reward }
    let rewardBound = Float(np.percentile(rewards, percentile))!
    print("rewardBound = \(rewardBound)")

    var input = Tensor<Float>(0.0)
    var target = Tensor<Float>(0.0)
    var totalReward: Float = 0.0

    var retainedEpisodeCount = 0
    for episode in episodes {
        if episode.reward < rewardBound {
            continue
        }

        let observationTensor = Tensor<Float>(episode.steps.map { $0.observation })
        let actionTensor = Tensor<Int32>(episode.steps.map { $0.action })
        let oneHotLabels = Tensor<Float>(oneHotAtIndices: actionTensor, depth: actionCount)

        // print("observations tensor has shape \(observationTensor.shapeTensor)")
        // print("actions tensor has shape \(actionTensor.shapeTensor)")
        // print("onehot actions tensor has shape \(oneHotLabels.shapeTensor)")

        if retainedEpisodeCount == 0 {
            input = observationTensor
            target = oneHotLabels
        } else {
            input = input.concatenated(with: observationTensor)
            target = target.concatenated(with: oneHotLabels)
        }
        // print("input tensor has shape \(input.shapeTensor)")
        // print("target tensor has shape \(target.shapeTensor)")

        totalReward += episode.reward
        retainedEpisodeCount += 1
    }

    return (input, target, retainedEpisodeCount, totalReward / Float(retainedEpisodeCount))
}

func nextBatch(
  env: PythonObject,
  net: Net,
  batchSize: Int,
  actionCount: Int
) -> [Episode] {
    var observationNumpy = env.reset()

    var episodes: [Episode] = []

    // We build up a batch of observations and actions.
    for _ in 0..<batchSize {
        var steps: [Episode.Step] = []
        var episodeReward: Float = 0.0

        while true {
            let observationPython = Tensor<Double>(numpy: observationNumpy).unwrapped()
            let actionProbabilities =
              softmax(net(Tensor(observationPython).reshaped(to: [1, 4])))
            let actionProbabilitiesPython = actionProbabilities[0].makeNumpyArray()
            let len = Python.len(actionProbabilitiesPython)
            assert(actionCount == Int(Python.len(actionProbabilitiesPython)))

            let actionPython = np.random.choice(len, p: actionProbabilitiesPython)
            let (nextObservation, reward, isDone, _) = env.step(actionPython).tuple4
            // print(nextObservation)
            // print(reward)

            steps.append(Episode.Step(observation: Tensor<Float>(observationPython),
                                      action: Int32(actionPython).unwrapped()))

            episodeReward += Float(reward).unwrapped()

            if isDone == true {
                // print("Finishing an episode with \(observations.count) steps and total reward \(episodeReward)")
                episodes.append(Episode(steps: steps, reward: episodeReward))
                observationNumpy = env.reset()
                break
            } else {
                observationNumpy = nextObservation
            }
        }
    }

    return episodes
}

let env = gym.make("CartPole-v0")
let observationSize = Int(env.observation_space.shape[0]).unwrapped()
let actionCount = Int(env.action_space.n).unwrapped()
// print(actionCount)

var net = Net(observationSize: Int(observationSize), hiddenSize: hiddenSize, actionCount: actionCount)
// SGD optimizer reaches convergence with ~125 mini batches, while Adam uses ~25.
// let optimizer = SGD<Net, Float>(learningRate: 0.1, momentum: 0.9)
let optimizer = Adam(for: net, learningRate: 0.01)
var batchIndex = 0

while true {
    print("Processing mini batch \(batchIndex)")
    batchIndex += 1

    let episodes = nextBatch(env: env, net: net, batchSize: batchSize, actionCount: actionCount)
    let (input, target, episodeCount, meanReward) = filteringBatch(
      episodes: episodes, actionCount: actionCount)

    let gradients = withLearningPhase(.training) {
        net.gradient { net -> Tensor<Float> in
            let logits = net(input)
            let loss = softmaxCrossEntropy(logits: logits, probabilities: target)
            print("loss is \(loss)")
            return loss
        }
    }
    optimizer.update(&net.allDifferentiableVariables, along: gradients)

    print("It has episode count \(episodeCount) and mean reward \(meanReward)")

    if meanReward > 199 {
        print("Solved")
        break
    }
}
