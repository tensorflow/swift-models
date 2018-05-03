import TensorFlow

// Note: This is a work in progress and training doesn't quite work.
// Here are areas for improvement:
// - Adopt a more principled reinforcement learning algorithm (e.g. policy
//   gradients). The algorithm should perform some tensor computation (not a
//   purely table-based approach).
// - `CatchAgent.step` calculates loss from the wrong reward. It uses the reward
//   at time `t+1`, but should actually use the reward from time `t`. This
//   requires saving the previous reward somehow.
// - The current back-propagation calculation may be incorrect.
// - It may be better to use a different initialization scheme for the layers of
//   `CatchAgent`.

extension Sequence {
  /// Returns elements' descriptions joined by a separator.
  func description(joinedBy separator: String) -> String {
    return map{"\($0)"}.joined(separator: separator)
  }
}

public typealias Observation = ShapedArray<Float>
public typealias Reward = Float

public protocol Environment {
  associatedtype Action : Equatable

  mutating func step(
    with action: Action
  ) -> (observation: Observation, reward: Reward)
  mutating func reset() -> Observation
}

public protocol Agent {
  associatedtype Action : Equatable

  mutating func step(
    with state: (observation: Observation, reward: Reward)
  ) -> Action
}

public struct CatchAgent : Agent {
  public typealias Action = CatchAction

  @_versioned var layer1: FullyConnectedLayer<Float>
  @_versioned var layer2: FullyConnectedLayer<Float>
  @_versioned let learningRate: Float
}

public extension CatchAgent {
  @_inlineable @inline(__always)
  init(learningRate: Float) {
    layer1 = FullyConnectedLayer(inputCount: 3, outputCount: 50)
    layer2 = FullyConnectedLayer(inputCount: 50, outputCount: 3)
    self.learningRate = learningRate
  }

  /// Performs one "step" (or parameter update) based on the specified
  /// observation and reward.
  @inline(never)
  mutating func step(
     with state: (observation: Observation, reward: Reward)
  ) -> Action {
    // NOTE: using `self.layer1` directly causes a send error. This is likely
    // because the function is mutating so referencing `self.layer1` produces a
    // load.
    // The workaround here is to:
    // - Bind `self.layer1` to a local variable.
    // - Perform tensor computations using the local variable.
    // - After all computations, set `self.layer1` to the local variable.

    // Initial setup.
    let (observation, reward) = state
    var layer1 = self.layer1
    var layer2 = self.layer2
    let learningRate = self.learningRate

    // Inference.
    let input = Tensor<Float>(observation).rankLifted()
    let pred1 = layer1.applied(to: input)
    let output1 = sigmoid(pred1)
    let pred2 = layer2.applied(to: output1)
    let output2 = sigmoid(pred2)
    let maxIndex = output2.argmax()

    // Back-propagation.
    let dOutput2 = output2 * (1 - output2)
    let (_, dParameters2) = layer2.gradient(for: output1,
                                            backpropagating: dOutput2)
    let dOutput1 = output1 * (1 - output1)
    let (_, dParameters1) = layer1.gradient(for: input,
                                            backpropagating: dOutput1)

    // Negative log loss.
    // FIXME: Loss is calculated from the wrong reward! It should be calculated
    // from the previous state. Fixing this is *most likely* to improve
    // training.
    // FIXME: indexing with `maxIndex` directly causes a send.
    // let loss = -log(output2[maxIndex]) * reward
    // FIXME: This infers to the variadic `max` function, which acts as a no-op.
    // let loss = -log(output2.max()) * reward
    let maxValue: Float = output2.max()
    let loss = -log(Tensor(maxValue)) * reward

    layer1.parameters.update(with: dParameters1,
                             by: { $0 -= learningRate * loss * $1 })
    layer2.parameters.update(with: dParameters2,
                             by: { $0 -= learningRate * loss * $1 })

    self.layer1 = layer1
    self.layer2 = layer2
    let action = CatchAction(rawValue: Int(maxIndex))!
    return action
  }

  /// Returns the perfect action, given an observation.
  /// If the ball is left of the paddle, returns `left`.
  /// If the ball is right of the paddle, returns `right`.
  /// Otherwise, returns `none`.
  /// Note: This function is for reference and is not used by `CatchAgent`.
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
    let id = Int(RandomState.global.generate()) % 3
    return CatchAction(rawValue: id)!
  }
}

public enum CatchAction : Int {
  case none
  case left
  case right
}

public struct Position : Equatable, Hashable {
  public var x: Int
  public var y: Int
}

public struct CatchEnvironment : Environment {
  public typealias Action = CatchAction
  public let rowCount: Int
  public let columnCount: Int
  public var ballPosition: Position
  public var paddlePosition: Position
}

public extension CatchEnvironment {
  init(rowCount: Int, columnCount: Int, seed: UInt32? = nil) {
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
    let currentReward = reward()
    // Return observation and reward.
    if ballPosition.y == rowCount {
      return (reset(), currentReward)
    }
    return (observation(), currentReward)
  }

  /// Resets the ball to be in a random column in the first row, and resets the
  /// paddle to be in the middle column of the bottom row.
  @discardableResult
  mutating func reset() -> Observation {
    let randomColumn = Int(RandomState.global.generate()) % columnCount
    ballPosition = Position(x: randomColumn, y: 0)
    paddlePosition = Position(x: columnCount / 2, y: rowCount - 1)
    return observation()
  }

  /// If the ball is in the bottom row:
  /// - Return 1 if the horizontal distance from the ball to the paddle is less
  ///   than or equal to 1.
  /// - Otherwise, return -1.
  /// If the ball is not in the bottom row, return 0.
  func reward() -> Float {
    if ballPosition.y == rowCount {
      return abs(ballPosition.x - paddlePosition.x) <= 1 ? 1 : -1
    }
    return 0
  }

  /// Returns an obeservation of the game grid.
  func observation() -> Observation {
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
    var result = ShapedArray<Float>(shape: [rowCount, columnCount], repeating: 0)
    result[ballPosition.y][ballPosition.x] = ShapedArraySlice(1 as Float)
    result[paddlePosition.y][paddlePosition.x] = ShapedArraySlice(1 as Float)
    return result
  }
}

extension CatchEnvironment : CustomStringConvertible {
  public var description: String {
    return grid.description(joinedBy: "\n")
  }
}

public func main() {
  // Set global seed.
  RandomState.global.seed(with: 42)

  // Setup environment and agent.
  var environment = CatchEnvironment(rowCount: 5, columnCount: 5)
  var action: CatchAction = .none
  var agent = CatchAgent(learningRate: 0.01)

  var gameCount = 0
  var winCount = 0
  var totalWinCount = 0
  let maxIterations = 1000
  repeat {
    // NOTE: the next line is the only one running tensor code.
    let state = environment.step(with: action)
    action = agent.step(with: state)

    if !state.reward.isZero {
      print("Game \(gameCount)", state.reward)
      gameCount += 1
      if state.reward > 0 {
        winCount += 1
        totalWinCount += 1
      }
      if gameCount % 20 == 0 {
        print("Win rate (last 20 games): \(Float(winCount) / 20)")
        print("""
          Win rate (total): \(Float(totalWinCount) / Float(gameCount)) \
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
}
main()
