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
    }
}
