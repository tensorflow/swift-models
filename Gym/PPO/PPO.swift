import TensorFlow


class PPO {
    let lr: Float
    let betas: [Float]
    let gamma: Float
    let K_epochs: Int
    let eps_clip: Float
    var actorCritic: ActorCritic
    var oldActorCritic: ActorCritic
    var actorOptimizer: Adam<ActorNet>
    var criticOptimizer: Adam<CriticNet>

    init(
        observationSize: Int,
        hiddenSize: Int,
        actionCount: Int,
        lr: Float,
        betas: [Float],
        gamma: Float,
        K_epochs: Int,
        eps_clip: Float
    ) {
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip

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
        self.actorOptimizer = Adam(for: actorCritic.actorNet, learningRate: lr)
        self.criticOptimizer = Adam(for: actorCritic.criticNet, learningRate: lr)

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

    func update(memory: Memory) {
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
        for _ in 0..<K_epochs {
            // Optimize policy network (actor)
            let (actorLoss, actorGradients) = valueWithGradient(at: self.actorCritic.actorNet) { actorNet -> Tensor<Float> in
                let npIndices = np.stack([np.arange(oldActions.shape[0], dtype: np.int32), oldActions.makeNumpyArray()], axis: 1)
                let tfIndices = Tensor<Int32>(numpy: npIndices)!
                let actionProbs = actorNet(oldStates).dimensionGathering(atIndices: tfIndices)

                let dist = Categorical<Int32>(probabilities: actionProbs)
                let stateValues = self.actorCritic.criticNet(oldStates).flattened()
                let ratios: Tensor<Float> = exp(dist.logProbabilities - oldLogProbs)

                let advantages: Tensor<Float> = tfRewards - stateValues
                let surrogateObjective1: Tensor<Float> = ratios * advantages
                let surrogateObjective2: Tensor<Float> = ratios.clipped(min:1 - self.eps_clip, max: 1 + self.eps_clip) * advantages
                let mainObjective = -1 * Tensor(stacking: [surrogateObjective1, surrogateObjective2]).min(alongAxes: 0).flattened()
                let entropyBonus: Tensor<Float> = -0.01 * Tensor<Float>(dist.entropy())
                let loss: Tensor<Float> = mainObjective + entropyBonus

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
