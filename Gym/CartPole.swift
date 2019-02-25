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

/// Model parameters and hyper parameters.
let hiddenSize = 128
let batchSize = 16
/// Controls the amount of good/long episodes to retain for training.
let percentile = 70

func scalarFromPython<Scalar: PythonConvertible>(value: PythonObject) -> Scalar {
  let swiftValue = Scalar.init(value)
  // Before force-unwrapping, we use `assert()` to make unwrapping nil more
  // debuggable (assert points to the source location.)
  assert(swiftValue != nil)
  return swiftValue!
}

func tensorFromPython<Scalar: NumpyScalarCompatible>(numpy: PythonObject) -> Tensor<Scalar> {
  let tensor = Tensor<Scalar>(numpy: numpy)
  // Before force-unwrapping, we use `assert()` to make unwrapping nil more
  // debuggable (assert points to the source location.)
  assert(tensor != nil)
  return tensor!
}

/// A simple two layer dense net.
struct Net: Layer {
  var l1, l2: Dense<Float>

  init(observationSize: Int, hiddenSize: Int, actionCount: Int) {
    self.l1 = Dense<Float>(
      inputSize: observationSize, outputSize: hiddenSize, activation: relu)

    self.l2 = Dense<Float>(
      inputSize: hiddenSize, outputSize: actionCount, activation: { $0 })
  }

  @differentiable(wrt: (self, input))
  func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
    return l2.applied(to: l1.applied(to: input, in: context), in: context)
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

/// Filter out bad/short episodes before we feed them as neural net training
/// data.
func filteringBatch(episodes: [Episode],
                    actionCount: Int32
) -> (input: Tensor<Float>,
      target: Tensor<Float>,
      episodeCount: Int,
      meanReward: Float) {
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

func nextBatch(env: PythonObject,
               net: Net,
               batchSize: Int,
               actionCount: Int32
) -> [Episode] {
  var obsNumpy = env.reset()

  let context = Context(learningPhase: .inference)
  var episodes: [Episode] = []
  
  // We build up a batch of observations and actions.
  for _ in 0..<batchSize {
    var steps: [Episode.Step] = []
    var episodeReward: Float = 0.0

    while true {
      let obsDouble: Tensor<Double> = tensorFromPython(numpy: obsNumpy)
      let obsFloat = Tensor<Float>(obsDouble).reshaped(to: [1, 4])
      let actionProbabilities = softmax(net.applied(to: obsFloat, in: context))

      let actProbs = actionProbabilities[0].makeNumpyArray()
      let len = Python.len(actProbs)
      assert(actionCount == scalarFromPython(value: len) as Int32)
      
      let actionNumpy = np.random.choice(len, p: actProbs)
      let (nextObs, reward, isDone, _) = env.step(actionNumpy).tuple4
      // print(nextObs)
      // print(reward)

      let observationDouble: Tensor<Double> = tensorFromPython(numpy: obsNumpy)
      let observationFloat = Tensor<Float>(observationDouble)
      let action: Int64 = scalarFromPython(value: actionNumpy)

      steps.append(Episode.Step(observation: observationFloat, action: Int32(action)))

      episodeReward += scalarFromPython(value: reward)

      if scalarFromPython(value: isDone) {
        // print("Finishing an episode with \(observations.count) steps and total reward \(episodeReward)")
        episodes.append(Episode(steps: steps, reward: episodeReward))
        obsNumpy = env.reset()
        break
      } else {
        obsNumpy = nextObs
      }
    }
  }

  return episodes
}

let env = gym.make("CartPole-v0")
let observationSize: Int32 = scalarFromPython(value: env.observation_space.shape[0])

let actionCount: Int32 = scalarFromPython(value: env.action_space.n)
// print(actionCount)

var net = Net(observationSize: Int(observationSize), hiddenSize: hiddenSize, actionCount: Int(actionCount))
/// SGD optimizer reaches convergence with ~125 mini batches, while Adam uses ~25.
// let optimizer = SGD<Net, Float>(learningRate: 0.1, momentum: 0.9)
let optimizer = Adam<Net, Float>(learningRate: 0.01)
var batchIndex = 0
while true {
  print("Processing mini batch \(batchIndex)")
  batchIndex += 1
  
  let episodes = nextBatch(env: env, net: net, batchSize: batchSize, actionCount: actionCount)
  let (input, target, episodeCount, meanReward) = filteringBatch(
    episodes: episodes, actionCount: actionCount)

  let gradients = gradient(at: net) { model -> Tensor<Float> in
    let context = Context(learningPhase: .training)
    let logits = model.applied(to: input, in: context)
    let loss = softmaxCrossEntropy(logits: logits, labels: target)
    print("loss is \(loss)")
    return loss
  }
  optimizer.update(&net.allDifferentiableVariables, along: gradients)

  print("It has episode count \(episodeCount) and mean reward \(meanReward)")

  if meanReward > 199 {
    print("Solved")
    break
  }
}
