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

import TensorFlow

// Force unwrapping with `!` does not provide source location when unwrapping `nil`, so we instead
// make a utility function for debuggability.
extension Optional {
  fileprivate func unwrapped(file: StaticString = #filePath, line: UInt = #line) -> Wrapped {
    guard let unwrapped = self else {
      fatalError("Value is nil", file: (file), line: line)
    }
    return unwrapped
  }
}

/// A Deep Q-Network.
///
/// A Q-network is a neural network that receives the observation (state) as input and estimates
/// the action values (Q values) of each action. For more information, check Human-level control
/// through deep reinforcement learning (Mnih et al., 2015).
struct DeepQNetwork: Layer {
  typealias Input = Tensor<Float>
  typealias Output = Tensor<Float>

  var l1, l2: Dense<Float>

  init(observationSize: Int, hiddenSize: Int, actionCount: Int) {
    l1 = Dense<Float>(inputSize: observationSize, outputSize: hiddenSize, activation: relu)
    l2 = Dense<Float>(inputSize: hiddenSize, outputSize: actionCount, activation: identity)
  }

  @differentiable
  func callAsFunction(_ input: Input) -> Output {
    return input.sequenced(through: l1, l2)
  }
}

/// Agent that uses the Deep Q-Network.
///
/// Deep Q-Network is an algorithm that trains a Q-network that estimates the action values of
/// each action given an observation (state). The Q-network is trained iteratively using the 
/// Bellman equation. For more information, check Human-level control through deep reinforcement
/// learning (Mnih et al., 2015).
class DeepQNetworkAgent {
  var qNet: DeepQNetwork
  var targetQNet: DeepQNetwork
  let optimizer: Adam<DeepQNetwork>
  let replayBuffer: ReplayBuffer
  let discount: Float
  let minBufferSize: Int
  let doubleDQN: Bool
  let device: Device

  init(
    qNet: DeepQNetwork,
    targetQNet: DeepQNetwork,
    optimizer: Adam<DeepQNetwork>,
    replayBuffer: ReplayBuffer,
    discount: Float,
    minBufferSize: Int,
    doubleDQN: Bool,
    device: Device
  ) {
    self.qNet = qNet
    self.targetQNet = targetQNet
    self.optimizer = optimizer
    self.replayBuffer = replayBuffer
    self.discount = discount
    self.minBufferSize = minBufferSize
    self.doubleDQN = doubleDQN
    self.device = device

    // Copy Q-network to Target Q-network before training
    updateTargetQNet(tau: 1)
  }

  func getAction(state: Tensor<Float>, epsilon: Float) -> Tensor<Int32> {
    if Float(np.random.uniform()).unwrapped() < epsilon {
      return Tensor<Int32>(numpy: np.array(np.random.randint(0, 2), dtype: np.int32))!
    } else {
      // Neural network input needs to be 2D
      let tfState = Tensor<Float>(numpy: np.expand_dims(state.makeNumpyArray(), axis: 0))!
      let qValues = qNet(tfState)[0]
      return Tensor<Int32>(qValues[1].scalarized() > qValues[0].scalarized() ? 1 : 0, on: device)
    }
  }

  func train(batchSize: Int) -> Float {
    // Don't train if replay buffer is too small
    if replayBuffer.count >= minBufferSize {
      let (tfStateBatch, tfActionBatch, tfRewardBatch, tfNextStateBatch, tfIsDoneBatch) =
        replayBuffer.sample(batchSize: batchSize)

      let (loss, gradients) = valueWithGradient(at: qNet) { qNet -> Tensor<Float> in
        // Compute prediction batch
        let npActionBatch = tfActionBatch.makeNumpyArray()
        let npFullIndices = np.stack(
          [np.arange(batchSize, dtype: np.int32), npActionBatch], axis: 1)
        let tfFullIndices = Tensor<Int32>(numpy: npFullIndices)!
        let stateQValueBatch = qNet(tfStateBatch)
        let predictionBatch = stateQValueBatch.dimensionGathering(atIndices: tfFullIndices)

        // Compute target batch
        let nextStateQValueBatch: Tensor<Float>
        if self.doubleDQN == true {
          // Double DQN
          let npNextStateActionBatch = self.qNet(tfNextStateBatch).argmax(squeezingAxis: 1)
            .makeNumpyArray()
          let npNextStateFullIndices = np.stack(
            [np.arange(batchSize, dtype: np.int32), npNextStateActionBatch], axis: 1)
          let tfNextStateFullIndices = Tensor<Int32>(numpy: npNextStateFullIndices)!
          nextStateQValueBatch = self.targetQNet(tfNextStateBatch).dimensionGathering(
            atIndices: tfNextStateFullIndices)
        } else {
          // DQN
          nextStateQValueBatch = self.targetQNet(tfNextStateBatch).max(squeezingAxes: 1)
        }
        let targetBatch: Tensor<Float> =
          tfRewardBatch + self.discount * (1 - Tensor<Float>(tfIsDoneBatch)) * nextStateQValueBatch

        return huberLoss(
          predicted: predictionBatch,
          expected: targetBatch,
          delta: 1
        )
      }
      optimizer.update(&qNet, along: gradients)

      return loss.scalarized()
    }
    return 0
  }

  func updateTargetQNet(tau: Float) {
    self.targetQNet.l1.weight =
      tau * Tensor<Float>(self.qNet.l1.weight) + (1 - tau) * self.targetQNet.l1.weight
    self.targetQNet.l1.bias =
      tau * Tensor<Float>(self.qNet.l1.bias) + (1 - tau) * self.targetQNet.l1.bias
    self.targetQNet.l2.weight =
      tau * Tensor<Float>(self.qNet.l2.weight) + (1 - tau) * self.targetQNet.l2.weight
    self.targetQNet.l2.bias =
      tau * Tensor<Float>(self.qNet.l2.bias) + (1 - tau) * self.targetQNet.l2.bias
  }
}
