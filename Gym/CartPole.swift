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

let HIDDEN_SIZE = 128
let BATCH_SIZE = 16
let PERCENTILE = 70

func scalarFromPython<Scalar: PythonConvertible>(value: PythonObject) -> Scalar {
  let swiftValue = Scalar.init(value)
  assert(swiftValue != nil)
  return swiftValue!
}

func tensorFromPython<Scalar: NumpyScalarCompatible>(numpy: PythonObject) -> Tensor<Scalar> {
  let tensor = Tensor<Scalar>(numpy: numpy)
  assert(tensor != nil)
  return tensor!
}

// A simple two layer dense net.
public struct Net: Layer {
  var l1, l2: Dense<Float>

  public init(obsSize: Int, hiddenSize: Int, actionCount: Int) {
    self.l1 = Dense<Float>(
      inputSize: obsSize, outputSize: hiddenSize, activation: { $0 })

    self.l2 = Dense<Float>(
      inputSize: hiddenSize, outputSize: actionCount, activation: { $0 })
  }

  @differentiable(wrt: (self, input))
  public func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
    return l2.applied(to: relu(l1.applied(to: input, in: context)), in: context)
  }
}

// An episode is a list of steps, where each step records the observation from
// env and the action taken. They will serve respectively as the input and
// target (label) of the neural net training.
// 
struct Episode {
  struct Step {
    let obs: Tensor<Float>
    let action: Int32
  }

  let steps: [Step]
  let reward: Float
}

func filterBatch(episodes: [Episode], actionCount: Int32
) -> (input: Tensor<Float>, target: Tensor<Float>, episodeCount: Int, meanReward: Float) {
  let rewards = episodes.map { $0.reward }
  let rewardBound = Float(np.percentile(rewards, PERCENTILE))!
  print("rewardBound = \(rewardBound)")

  var input = Tensor<Float>(0.0)
  var target = Tensor<Float>(0.0)
  var totalReward: Float = 0.0

  var retainedEpisodeCount = 0
  for episode in episodes {
    if episode.reward < rewardBound {
      continue
    }

    let obsTensor = Tensor<Float>(episode.steps.map { $0.obs })
    let actsTensor = Tensor<Int32>(episode.steps.map { $0.action })
    let oneHotLabels = Tensor<Float>(oneHotAtIndices: actsTensor, depth: actionCount)

    // print("observations tensor has shape \(obsTensor.shapeTensor)")
    // print("actions tensor has shape \(actsTensor.shapeTensor)")
    // print("onehot actions tensor has shape \(oneHotLabels.shapeTensor)")

    if retainedEpisodeCount == 0 {
      input = obsTensor
      target = oneHotLabels
    } else {
      input = input.concatenated(with: obsTensor)
      target = target.concatenated(with: oneHotLabels)
    }
    // print("input tensor has shape \(input.shapeTensor)")
    // print("target tensor has shape \(target.shapeTensor)")

    totalReward += episode.reward
    retainedEpisodeCount += 1
  }
  
  return (input, target, retainedEpisodeCount, totalReward / Float(retainedEpisodeCount))
}

// We pass in `net` by reference to avoid making a copy, even though this function is
// running inferencing over net, and not mutating it.
func nextBatch(env: PythonObject, net: inout Net, batchSize: Int, actionCount: Int32
) -> [Episode] {
  var obsNumpy = env.reset()

  let context = Context(learningPhase: .inference)
  var episodes: [Episode] = []
  
  // We build up a batch of observations and actions
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

      steps.append(Episode.Step(obs: observationFloat, action: Int32(action)))

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

func loss(model: Net, input: Tensor<Float>, target: Tensor<Float>)
  -> Tensor<Float> {
  let context = Context(learningPhase: .training)
  let logits = model.applied(to: input, in: context)
  return softmaxCrossEntropy(logits: logits, labels: target)
}

func train() {
  let env = gym.make("CartPole-v0")
  let obsSize: Int32 = scalarFromPython(value: env.observation_space.shape[0])

  let actionCount: Int32 = scalarFromPython(value: env.action_space.n)
  // print(actionCount)

  var net = Net(obsSize: Int(obsSize), hiddenSize: HIDDEN_SIZE, actionCount: Int(actionCount))
  let optimizer = SGD<Net, Float>(learningRate: 0.1, momentum: 0.9)
  var batchIndex = 0
  while true {
    print("Processing mini batch \(batchIndex)")
    batchIndex += 1
    
    let episodes = nextBatch(env: env, net: &net, batchSize: BATCH_SIZE, actionCount: actionCount)
    let (input, target, episodeCount, meanReward) = filterBatch(
      episodes: episodes, actionCount: actionCount)

    let gradients = gradient(at: net) { model -> Tensor<Float> in
      let l = loss(model: model, input: input, target: target)
      print("loss is \(l)")
      return l
    }
    optimizer.update(&net.allDifferentiableVariables, along: gradients)

    print("It has episode count \(episodeCount) and mean reward \(meanReward)")

    if meanReward > 199 {
      print("Solved")
      break
    }
  }
}

train()
