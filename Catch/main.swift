// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

// Note: This is a work in progress and training doesn't quite work.
// Here are areas for improvement:
// - Adopt a more principled reinforcement learning algorithm (e.g. policy
//   gradients). The algorithm should perform some tensor computation (not a
//   purely table-based approach).

var rng = PhiloxRandomNumberGenerator(seed: 0xdeadbeef)

extension Sequence {
    /// Returns elements' descriptions joined by a separator.
    func description(joinedBy separator: String) -> String {
        return map{"\($0)"}.joined(separator: separator)
    }
}

typealias Observation = ShapedArray<Float>
typealias Reward = Float

protocol Environment {
    associatedtype Action: Equatable
    mutating func step(with action: Action) -> (observation: Observation, reward: Reward)
    mutating func reset() -> Observation
}

protocol Agent: AnyObject {
    associatedtype Action: Equatable
    func step(observation: Observation, reward: Reward) -> Action
}

class CatchAgent: Agent {
    typealias Action = CatchAction

    var model = Sequential {
        Dense<Float>(inputSize: 3, outputSize: 50, activation: sigmoid, generator: &rng)
        Dense<Float>(inputSize: 50, outputSize: 3, activation: sigmoid, generator: &rng)
    }

    let optimizer: Adam<Model>
    var previousReward: Reward

    init(initialReward: Reward, learningRate: Float) {
        optimizer = Adam(for: model, learningRate: learningRate)
        previousReward = initialReward
    }
}

extension CatchAgent {
    /// Performs one "step" (or parameter update) based on the specified
    /// observation and reward.
    func step(
        observation: Observation, reward: Reward
    ) -> Action {
        defer { previousReward = reward }

        let x = Tensor(observation).rankLifted()
        let (ŷ, backprop) = model.appliedForBackpropagation(to: x)
        let maxIndex = ŷ.argmax().scalarized()

        let 𝛁loss = -log(Tensor<Float>(ŷ.max())).broadcasted(like: ŷ) * previousReward
        let (𝛁model, _) = backprop(𝛁loss)
        optimizer.update(&model, along: 𝛁model)

        return CatchAction(rawValue: Int(maxIndex))!
    }

    /// Returns the perfect action, given an observation.
    /// If the ball is left of the paddle, returns `left`.
    /// If the ball is right of the paddle, returns `right`.
    /// Otherwise, returns `none`.
    ///
    /// - Note: This function is for reference and is not used by `CatchAgent`.
    func perfectAction(for observation: Observation) -> Action {
        let paddleX = observation.scalars[0]
        let ballX = observation.scalars[1]
        if paddleX > ballX {
            return .right
        } else if paddleX < ballX {
            return .left
        }
        return .none
    }

    /// Returns a random action.
    /// Note: This function is for reference and is not used by `CatchAgent`.
    func randomAction() -> Action {
        let id = Int.random(in: 0..<3, using: &rng)
        return CatchAction(rawValue: id)!
    }
}

enum CatchAction: Int {
    case none
    case left
    case right
}

struct Position: Equatable, Hashable {
    var x: Int
    var y: Int
}

struct CatchEnvironment: Environment {
    typealias Action = CatchAction
    let rowCount: Int
    let columnCount: Int
    var ballPosition: Position
    var paddlePosition: Position
}

extension CatchEnvironment {
    init(rowCount: Int, columnCount: Int) {
        self.rowCount = rowCount
        self.columnCount = columnCount
        self.ballPosition = Position(x: 0, y: 0)
        self.paddlePosition = Position(x: 0, y: 0)
        reset()
    }

    mutating func step(
        with action: CatchAction
    ) -> (observation: Observation, reward: Float) {
        // Update state.
        switch action {
        case .left where paddlePosition.x > 0:
            paddlePosition.x -= 1
        case .right where paddlePosition.x < columnCount - 1:
            paddlePosition.x += 1
        default:
            break
        }
        ballPosition.y += 1
        // Get reward.
        let currentReward = reward
        // Return observation and reward.
        if ballPosition.y == rowCount {
            return (reset(), currentReward)
        }
        return (observation, currentReward)
    }

    /// Resets the ball to be in a random column in the first row, and resets
    /// the paddle to be in the middle column of the bottom row.
    @discardableResult
    mutating func reset() -> Observation {
        let randomColumn = Int.random(in: 0..<columnCount, using: &rng)
        ballPosition = Position(x: randomColumn, y: 0)
        paddlePosition = Position(x: columnCount / 2, y: rowCount - 1)
        return observation
    }

    /// If the ball is in the bottom row:
    /// - Returns 1 if the horizontal distance from the ball to the paddle is
    ///   less than or equal to 1.
    /// - Otherwise, returns -1.
    /// If the ball is not in the bottom row, returns 0.
    var reward: Float {
        if ballPosition.y == rowCount {
            return abs(ballPosition.x - paddlePosition.x) <= 1 ? 1 : -1
        }
        return 0
    }

    /// Returns an obeservation of the game grid.
    var observation: Observation {
        return ShapedArray<Float>(
            shape: [3],
            scalars: [Float(ballPosition.x) / Float(columnCount),
                      Float(ballPosition.y) / Float(rowCount),
                      Float(paddlePosition.x) / Float(columnCount)]
        )
    }

    /// Returns the game grid as a 2D matrix where all scalars are 0 except the
    /// positions of the ball and paddle, which are 1.
    var grid: ShapedArray<Float> {
        var result = ShapedArray<Float>(repeating: 0, shape: [rowCount, columnCount])
        result[ballPosition.y][ballPosition.x] = ShapedArraySlice<Float>(1)
        result[paddlePosition.y][paddlePosition.x] = ShapedArraySlice<Float>(1)
        return result
    }
}

extension CatchEnvironment: CustomStringConvertible {
    var description: String {
        return grid.description(joinedBy: "\n")
    }
}

// Setup environment and agent.
Context.local.learningPhase = .training
var environment = CatchEnvironment(rowCount: 5, columnCount: 5)
var action: CatchAction = .none
var agent = CatchAgent(initialReward: environment.reward, learningRate: 0.05)

var gameCount = 0
var winCount = 0
var totalWinCount = 0
let maxIterations = 5000
repeat {
    let (observation, reward) = environment.step(with: action)
    action = agent.step(observation: observation, reward: reward)

    if !reward.isZero {
        gameCount += 1
        if reward > 0 {
            winCount += 1
            totalWinCount += 1
        }
        if gameCount % 20 == 0 {
            print("Win rate (last 20 games): \(Float(winCount) / 20)")
            print("""
                  Win rate (total): \
                  \(Float(totalWinCount) / Float(gameCount)) \
                  [\(totalWinCount)/\(gameCount)]
                  """)
            winCount = 0
        }
    }
} while gameCount < maxIterations

print("""
      Win rate (final): \(Float(totalWinCount) / Float(gameCount)) \
      [\(totalWinCount)/\(gameCount)]
      """)
