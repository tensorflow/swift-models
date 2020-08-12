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
let hiddenSize: Int = 64
// Optimizer HP
let lr: Float = 0.002
let betas: [Float] = [0.9, 0.999]
let gamma: Float = 0.99
let K_epochs: Int = 4
let eps_clip: Float = 0.2
// Interaction
let maxEpisodes: Int = 500
let maxTimesteps: Int = 300
let updateTimestep: Int = 2000
// Log
let logInterval: Int = 20
let solvedReward: Float = 199

var memory: Memory = Memory()
var ppo = PPO(
    observationSize: observationSize,
    hiddenSize: hiddenSize,
    actionCount: actionCount,
    lr: lr,
    betas: betas,
    gamma: gamma,
    K_epochs: K_epochs,
    eps_clip: eps_clip
)

// Training loop
var timestep: Int = 0
var runningReward: Float = 0
var averageLength: Float = 0
var episodeReturn: Float = 0
var episodeReturns: [Float] = []
var bestEpisodeReturn: Float = 0
for episodeIndex in 1..<maxEpisodes+1 {
    var state = env.reset()
    var t: Int = 1
    for _ in 0..<maxTimesteps {
        timestep += 1
        let tfState = Tensor<Float>(numpy: np.array(state, dtype: np.float32))!
        let action: Int32 = ppo.oldActorCritic.act(state: tfState, memory: memory)
        let (newState, reward, done, _) = env.step(action).tuple4
        memory.rewards.append(Float(reward)!)
        memory.isDones.append(Bool(done)!)

        if timestep % updateTimestep == 0 {
            ppo.update(memory: memory)
            memory.clear_memory()
            timestep = 0
        }

        runningReward += Float(reward)!
        episodeReturn += Float(reward)!
        if Bool(done)! == true {
            if bestEpisodeReturn < episodeReturn {
                print("Return: \(episodeReturn)")
                bestEpisodeReturn = episodeReturn
            }
            episodeReturns.append(episodeReturn)
            episodeReturn = 0
            break
        }

        t += 1
        state = newState
    }

    averageLength += Float(t)

    if Float(runningReward) > (Float(logInterval) * Float(solvedReward)) {
        print("########## Solved! ##########")
        break
    }

    if episodeIndex % logInterval == 0 {
        averageLength = Float(averageLength) / Float(logInterval)
        runningReward = Float(runningReward) / Float(logInterval)

        print(String(format: "Episode: %4d | Average Length %5.2f | Running Reward: %5.2f", episodeIndex, averageLength, runningReward))
    }
}

// Save learning curve
plt.plot(episodeReturns)
plt.title("Proximal Policy Optimization on CartPole-v0")
plt.xlabel("Episode")
plt.ylabel("Episode Return")
plt.savefig("ppoEpisodeReturns.png")
plt.clf()
