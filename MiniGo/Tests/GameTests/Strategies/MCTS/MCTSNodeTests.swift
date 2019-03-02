import TensorFlow
import XCTest

@testable import Game

final class MCTSNodeTests: XCTestCase {
    func testSelectActionWithHighestPriorForNewNode() {
        let boardSize = 2
        let configuration = GameConfiguration(size: boardSize, komi: 0.1)
        let boardState = BoardState(gameConfiguration: configuration)

        let node = MCTSNode(
            boardSize: boardSize,
            boardState: boardState,
            distribution: MCTSPrediction.Distribution(
                positions: ShapedArray<Float>(
                    shape: [boardSize, boardSize],
                    scalars: Array(1...boardSize * boardSize).map { Float($0) }),
                pass: 0.0))

        // The final position has the highest prior.
        let action = node.actionByPUCT
        XCTAssertEqual(.place(position: Position(x: boardSize-1, y: boardSize-1)), action)
    }

    func testSelectActionWithHighestPriorWithTieBreakForNewNode() {
        let boardSize = 2
        let configuration = GameConfiguration(size: boardSize, komi: 0.1)
        let boardState = BoardState(gameConfiguration: configuration)

        let node = MCTSNode(
            boardSize: boardSize,
            boardState: boardState,
            distribution: MCTSPrediction.Distribution(
                positions: ShapedArray<Float>(shape: [boardSize, boardSize], repeating: 1.0),
                pass: 1.0))

        // All actions have equal prior values. So, with 100 runs, we should see they are randomly
        // selected. To avoid flakyness, we test >1 rather than ==boardSize*boardSize+1.
        var actionPool = Set<Move>()
        for _ in 1..<100 {
            actionPool.insert(node.actionByPUCT)
        }
        XCTAssertTrue(actionPool.count > 1)
    }
}

// Tests for backUp.
extension MCTSNodeTests {
    func testSelectActionAfterBackUpForSamePlayer() {
        let boardSize = 2
        let configuration = GameConfiguration(size: boardSize, komi: 0.1)
        let boardState = BoardState(gameConfiguration: configuration)
        XCTAssertEqual(.black, boardState.nextPlayerColor)

        let node = MCTSNode(
            boardSize: boardSize,
            boardState: boardState,
            distribution: MCTSPrediction.Distribution(
                positions: ShapedArray<Float>(shape: [boardSize, boardSize], repeating: 1.0),
                pass: 1.0))

        node.backUp(for: .place(position: Position(x: 1, y: 0)), withRewardForBlackPlayer: 100.0)
        let action = node.actionByPUCT
        XCTAssertEqual(.place(position: Position(x: 1, y: 0)), action)
    }

    func testSelectActionAfterBackUpForOpponent() {
        let boardSize = 2
        let configuration = GameConfiguration(size: boardSize, komi: 0.1)
        let boardState = BoardState(gameConfiguration: configuration).passing()
        XCTAssertEqual(.white, boardState.nextPlayerColor)

        let node = MCTSNode(
            boardSize: boardSize,
            boardState: boardState,
            distribution: MCTSPrediction.Distribution(
                positions: ShapedArray<Float>(
                    shape: [boardSize, boardSize],
                    scalars: Array(1...boardSize * boardSize).map { Float($0) }),
                pass: Float(boardSize * boardSize) - 0.5))

        // x: 1, y: 1 has the highest prior. But after backing up, its action value is lower.
        node.backUp(for: .place(position: Position(x: 1, y: 1)), withRewardForBlackPlayer: 100.0)
        let action = node.actionByPUCT
        XCTAssertEqual(.pass, action)
    }
}

extension MCTSNodeTests {
    static var allTests = [
        ("testSelectActionWithHighestPriorForNewNode", testSelectActionWithHighestPriorForNewNode),
        ("testSelectActionWithHighestPriorWithTieBreakForNewNode",
         testSelectActionWithHighestPriorWithTieBreakForNewNode),
        ("testSelectActionAfterBackUpForSamePlayer", testSelectActionAfterBackUpForSamePlayer),
        ("testSelectActionAfterBackUpForOpponent", testSelectActionAfterBackUpForOpponent),
    ]
}
