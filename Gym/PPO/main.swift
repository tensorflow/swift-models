import PythonKit
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


let env = gym.make("CartPole-v0")
let observationSize: Int = Int(env.observation_space.shape[0])!
let actionCount: Int = Int(env.action_space.n)!

// Hyperparameters
// Network HP
let hiddenSize: Int = 128
// Optimizer HP
let learningRate: Float = 0.0003
// TODO(seungjaeryanlee): Not used
let betas: [Float] = [0.9, 0.999]
let gamma: Float = 0.99
let epochs: Int = 10
let clipEpsilon: Float = 0.1
let entropyCoefficient: Float = 0.0001
// Interaction
let maxEpisodes: Int = 1000
let maxTimesteps: Int = 200
let updateTimestep: Int = 1000
// Log
let logInterval: Int = 20
let solvedReward: Float = 199

var memory: PPOMemory = PPOMemory()
var agent: PPOAgent = PPOAgent(
    observationSize: observationSize,
    hiddenSize: hiddenSize,
    actionCount: actionCount,
    learningRate: learningRate,
    betas: betas,
    gamma: gamma,
    epochs: epochs,
    clipEpsilon: clipEpsilon,
    entropyCoefficient: entropyCoefficient
)

// Training loop
var timestep: Int = 0
var episodeReturn: Float = 0
var episodeReturns: [Float] = []
var averageReturn: Float = 0
for episodeIndex in 1..<maxEpisodes+1 {
    var state = env.reset()
    for _ in 0..<maxTimesteps {
        timestep += 1
        let tfState = Tensor<Float>(numpy: np.array(state, dtype: np.float32))!
        let action: Int32 = agent.oldActorCritic.act(state: tfState, memory: memory)
        let (newState, reward, done, _) = env.step(action).tuple4
        memory.rewards.append(Float(reward)!)
        memory.isDones.append(Bool(done)!)

        if timestep % updateTimestep == 0 {
            agent.update(memory: memory)
            memory.clear_memory()
            timestep = 0
        }

        averageReturn += Float(reward)!
        episodeReturn += Float(reward)!
        if Bool(done)! == true {
            episodeReturns.append(episodeReturn)
            episodeReturn = 0
            break
        }

        state = newState
    }

    if episodeIndex % logInterval == 0 {
        averageReturn = Float(averageReturn) / Float(logInterval)
        print(String(format: "Episode: %4d | Average Return: %6.2f", episodeIndex, averageReturn))
    }

    if Float(averageReturn) > (Float(logInterval) * Float(solvedReward)) {
        break
    }
}

// Save learning curve
plt.plot(episodeReturns)
plt.title("Proximal Policy Optimization on CartPole-v0")
plt.xlabel("Episode")
plt.ylabel("Episode Return")
plt.savefig("ppoEpisodeReturns.png")
plt.clf()
