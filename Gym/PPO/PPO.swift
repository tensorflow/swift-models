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

    init(observationSize: Int, hiddenSize: Int, actionCount: Int, lr: Float, betas: [Float], gamma: Float, K_epochs: Int, eps_clip: Float) {
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
        // TODO: Copy actorCritic to oldActorCritic
        self.actorOptimizer = Adam(for: actorCritic.actorNet, learningRate: lr)
        self.criticOptimizer = Adam(for: actorCritic.criticNet, learningRate: lr)
    }

    func update(memory: Memory) {
        // TODO: Implement
        var rewards: [Float] = []
        var discounted_reward: Float = 0
        for i in (0..<memory.rewards.count).reversed() {
            if memory.isDones[i] {
                discounted_reward = 0
            }
            discounted_reward = memory.rewards[i] + (gamma * discounted_reward)
            rewards.insert(discounted_reward, at: 0)
        }
        var tfRewards = Tensor<Float>(rewards)
        tfRewards = (tfRewards - tfRewards.mean()) / (tfRewards.standardDeviation() + 1e-5)

        let old_states: Tensor<Float> = Tensor<Float>(numpy: np.array(memory.states, dtype: np.float32))!
        let old_actions: Tensor<Int32> = Tensor<Int32>(numpy: np.array(memory.actions, dtype: np.int32))!
        let old_logprobs: Tensor<Float> = Tensor<Float>(numpy: np.array(memory.logProbs, dtype: np.float32))!

        // TODO: Optimize both critic and actor
        var net = self.actorCritic.criticNet
        let optimizer = Adam(for: net, learningRate: lr)

        for _ in 0..<K_epochs {
            let (_, gradients) = valueWithGradient(at: net) { net -> Tensor<Float> in
                let (logProbs, state_values, dist_entropy) = self.actorCritic.evaluate(state: old_states, action: old_actions)
                let ratios: Tensor<Float> = exp(logProbs - old_logprobs)

                let advantages: Tensor<Float> = tfRewards - state_values
                let surr1: Tensor<Float> = ratios * advantages
                let surr2: Tensor<Float> = ratios.clipped(min:1 - self.eps_clip, max: 1 + self.eps_clip) * advantages
                // TODO(seungjaeryanlee): Find how to find minimum
                // let loss1: PythonObject =  -1 * np.minimum(surr1.makeNumpyArray(), surr2.makeNumpyArray())
                let loss1: Tensor<Float> = 0 * surr1 + -1 * surr2
                let loss2: Tensor<Float> = 0.5 * pow(state_values - tfRewards, 2)
                let loss3: Tensor<Float> = Tensor<Float>(-0.01 * dist_entropy)
                let loss: Tensor<Float> = loss1 + loss2 + loss3

                return loss.mean()
            }
            // TODO: Fix gradients all being zero
            // print(loss)
            // print(gradients)

            optimizer.update(&net, along: gradients)
        }
        self.oldActorCritic.criticNet.l1.weight = self.actorCritic.criticNet.l1.weight
        self.oldActorCritic.criticNet.l1.bias = self.actorCritic.criticNet.l1.bias
        self.oldActorCritic.criticNet.l2.weight = self.actorCritic.criticNet.l2.weight
        self.oldActorCritic.criticNet.l2.bias = self.actorCritic.criticNet.l2.bias
        self.oldActorCritic.actorNet.l1.weight = self.actorCritic.actorNet.l1.weight
        self.oldActorCritic.actorNet.l1.bias = self.actorCritic.actorNet.l1.bias
        self.oldActorCritic.actorNet.l2.weight = self.actorCritic.actorNet.l2.weight
        self.oldActorCritic.actorNet.l2.bias = self.actorCritic.actorNet.l2.bias
    }
}
