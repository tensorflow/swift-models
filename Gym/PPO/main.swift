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
let max_episodes: Int = 500
let max_timesteps: Int = 300
let update_timestep: Int = 2000
// Log
let log_interval: Int = 20
let solved_reward: Float = 199

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
var running_reward: Float = 0
var avg_length: Float = 0
for i_episode in 1..<max_episodes+1 {
    var state = env.reset()
    var t: Int = 0
    for t in 0..<max_timesteps {
        timestep += 1
        let action: Int32 = ppo.oldActorCritic.act(state: Tensor<Float>(numpy: np.array(state, dtype: np.float32))!, memory: memory)
        var (state, reward, done, _) = env.step(action).tuple4

        memory.rewards.append(Float(reward)!)
        memory.isDones.append(Bool(done)!)

        if timestep % update_timestep == 0 {
            ppo.update(memory: memory)
            memory.clear_memory()
            timestep = 0
        }

        running_reward += Float(reward)!
        if Bool(done)! == true {
            break
        }
    }

    avg_length += Float(t)

    if Float(running_reward) > (Float(log_interval) * Float(solved_reward)) {
        print("########## Solved! ##########")
        break
    }

    if i_episode % log_interval == 0 {
        avg_length = Float(avg_length) / Float(log_interval)
        running_reward = Float(running_reward) / Float(log_interval)

        print(String(format: "Episode: %4d | Average Length %5.2f | Running Reward: %5.2f", i_episode, avg_length, running_reward))
    }
}
