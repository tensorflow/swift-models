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

class PPOAgent {
    let learningRate: Float
    let betas: [Float]
    let gamma: Float
    let epochs: Int
    let clipEpsilon: Float
    let entropyCoefficient: Float
    var actorCritic: ActorCritic
    var oldActorCritic: ActorCritic
    var actorOptimizer: Adam<ActorNet>
    var criticOptimizer: Adam<CriticNet>

    init(
        observationSize: Int,
        hiddenSize: Int,
        actionCount: Int,
        learningRate: Float,
        betas: [Float],
        gamma: Float,
        epochs: Int,
        clipEpsilon: Float,
        entropyCoefficient: Float
    ) {
        self.learningRate = learningRate
        self.betas = betas
        self.gamma = gamma
        self.epochs = epochs
        self.clipEpsilon = clipEpsilon
        self.entropyCoefficient = entropyCoefficient

        self.actorCritic = ActorCritic(
            observationSize: observationSize,
            hiddenSize: hiddenSize,
            actionCount: actionCount
        )
        self.oldActorCritic = ActorCritic(
            observationSize: observationSize,
            hiddenSize: hiddenSize,
            actionCount: actionCount
        )
        self.actorOptimizer = Adam(for: actorCritic.actorNet, learningRate: learningRate)
        self.criticOptimizer = Adam(for: actorCritic.criticNet, learningRate: learningRate)

        // Copy actorCritic to oldActorCritic
        self.updateOldActorCritic()
    }

    func updateOldActorCritic() {
        self.oldActorCritic.criticNet.l1.weight = self.actorCritic.criticNet.l1.weight
        self.oldActorCritic.criticNet.l1.bias = self.actorCritic.criticNet.l1.bias
        self.oldActorCritic.criticNet.l2.weight = self.actorCritic.criticNet.l2.weight
        self.oldActorCritic.criticNet.l2.bias = self.actorCritic.criticNet.l2.bias
        self.oldActorCritic.criticNet.l3.weight = self.actorCritic.criticNet.l3.weight
        self.oldActorCritic.criticNet.l3.bias = self.actorCritic.criticNet.l3.bias
        self.oldActorCritic.actorNet.l1.weight = self.actorCritic.actorNet.l1.weight
        self.oldActorCritic.actorNet.l1.bias = self.actorCritic.actorNet.l1.bias
        self.oldActorCritic.actorNet.l2.weight = self.actorCritic.actorNet.l2.weight
        self.oldActorCritic.actorNet.l2.bias = self.actorCritic.actorNet.l2.bias
        self.oldActorCritic.actorNet.l3.weight = self.actorCritic.actorNet.l3.weight
        self.oldActorCritic.actorNet.l3.bias = self.actorCritic.actorNet.l3.bias
    }

    func update(memory: PPOMemory) {
        // Discount rewards for advantage estimation
        var rewards: [Float] = []
        var discountedReward: Float = 0
        for i in (0..<memory.rewards.count).reversed() {
            if memory.isDones[i] {
                discountedReward = 0
            }
            discountedReward = memory.rewards[i] + (gamma * discountedReward)
            rewards.insert(discountedReward, at: 0)
        }
        var tfRewards = Tensor<Float>(rewards)
        tfRewards = (tfRewards - tfRewards.mean()) / (tfRewards.standardDeviation() + 1e-5)

        // Retrieve stored states, actions, and log probabilities
        let oldStates: Tensor<Float> = Tensor<Float>(numpy: np.array(memory.states, dtype: np.float32))!
        let oldActions: Tensor<Int32> = Tensor<Int32>(numpy: np.array(memory.actions, dtype: np.int32))!
        let oldLogProbs: Tensor<Float> = Tensor<Float>(numpy: np.array(memory.logProbs, dtype: np.float32))!

        // Optimize actor and critic
        var actorLosses: [Float] = []
        var criticLosses: [Float] = []
        for _ in 0..<epochs {
            // Optimize policy network (actor)
            let (actorLoss, actorGradients) = valueWithGradient(at: self.actorCritic.actorNet) { actorNet -> Tensor<Float> in
                let npIndices = np.stack([np.arange(oldActions.shape[0], dtype: np.int32), oldActions.makeNumpyArray()], axis: 1)
                let tfIndices = Tensor<Int32>(numpy: npIndices)!
                let actionProbs = actorNet(oldStates).dimensionGathering(atIndices: tfIndices)

                let dist = Categorical<Int32>(probabilities: actionProbs)
                let stateValues = self.actorCritic.criticNet(oldStates).flattened()
                let ratios: Tensor<Float> = exp(dist.logProbabilities - oldLogProbs)

                let advantages: Tensor<Float> = tfRewards - stateValues
                let surrogateObjective = Tensor(stacking: [
                    ratios * advantages,
                    ratios.clipped(min:1 - self.clipEpsilon, max: 1 + self.clipEpsilon) * advantages
                ]).min(alongAxes: 0).flattened()
                let entropyBonus: Tensor<Float> = Tensor<Float>(self.entropyCoefficient * dist.entropy())
                let loss: Tensor<Float> = -1 * (surrogateObjective + entropyBonus)

                return loss.mean()
            }
            self.actorOptimizer.update(&self.actorCritic.actorNet, along: actorGradients)
            actorLosses.append(actorLoss.scalarized())

            // Optimize value network (critic)
            let (criticLoss, criticGradients) = valueWithGradient(at: self.actorCritic.criticNet) { criticNet -> Tensor<Float> in
                let stateValues = criticNet(oldStates).flattened()
                let loss: Tensor<Float> = 0.5 * pow(stateValues - tfRewards, 2)

                return loss.mean()
            }
            self.criticOptimizer.update(&self.actorCritic.criticNet, along: criticGradients)
            criticLosses.append(criticLoss.scalarized())
        }
        self.updateOldActorCritic()
    }
}
