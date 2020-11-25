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

import XCTest

@testable import MiniGo

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
