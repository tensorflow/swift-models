import XCTest

@testable import Game

final class MCTSPolicyTests: XCTestCase {
    func testChooseReasonableMove() {
        let boardSize = 2
        let configuration = GameConfiguration(size: 2, komi: 100.0)
        var boardState = BoardState(gameConfiguration: configuration)

        boardState = boardState.passing()
        XCTAssertEqual(.white, boardState.nextPlayerColor)

        let policy = MCTSPolicy(
            participantName: "mcts",
            predictor: MCTSRandomPredictor(boardSize: boardSize),
            configuration: MCTSConfiguration(gameConfiguration: configuration))

        // The white player has super big advantage due to large komi. Even using a random prediction,
        // the MCTS should discover this and choose .pass to end the game
        let nextMove = policy.nextMove(for: boardState, after: .pass)
        XCTAssertEqual(.pass, nextMove)
    }
}

extension MCTSPolicyTests {
    static var allTests = [
        ("testChooseReasonableMove", testChooseReasonableMove),
    ]
}
