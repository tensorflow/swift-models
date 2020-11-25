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

final class BoardStateTests: XCTestCase {
    func testEmptyBoard() throws {
        let configuration = GameConfiguration(size: 2, komi: 0.1)
        let boardState = BoardState(gameConfiguration: configuration)
        XCTAssertEqual(.black, boardState.nextPlayerColor)
        XCTAssertNil(boardState.ko)
        XCTAssertEqual(0, boardState.playedMoveCount)
        XCTAssertEqual(0, boardState.stoneCount)
        let expectedLegalMoves = [
            Position(x: 0, y: 0),
            Position(x: 0, y: 1),
            Position(x: 1, y: 0),
            Position(x: 1, y: 1),
        ]
        XCTAssertEqual(expectedLegalMoves, boardState.legalMoves)
    }

    func testPassing() throws {
        let configuration = GameConfiguration(size: 2, komi: 0.1)
        let boardState = BoardState(gameConfiguration: configuration).passing()
        // The following two fields are different.
        XCTAssertEqual(.white, boardState.nextPlayerColor)
        XCTAssertEqual(1, boardState.playedMoveCount)

        // The rest is same as before passing.
        XCTAssertNil(boardState.ko)
        XCTAssertEqual(0, boardState.stoneCount)
        let expectedLegalMoves = [
            Position(x: 0, y: 0),
            Position(x: 0, y: 1),
            Position(x: 1, y: 0),
            Position(x: 1, y: 1),
        ]
        XCTAssertEqual(expectedLegalMoves, boardState.legalMoves)
    }

    func testPlacingNewStone() throws {
        let configuration = GameConfiguration(size: 2, komi: 0.1)
        var boardState = BoardState(gameConfiguration: configuration)
        boardState = try boardState.placingNewStone(at: Position(x: 0, y: 0))

        XCTAssertEqual(.white, boardState.nextPlayerColor)
        XCTAssertNil(boardState.ko)
        XCTAssertEqual(1, boardState.playedMoveCount)
        XCTAssertEqual(1, boardState.stoneCount)
        let expectedLegalMoves = [
            Position(x: 0, y: 1),
            Position(x: 1, y: 0),
            Position(x: 1, y: 1),
        ]
        XCTAssertEqual(expectedLegalMoves, boardState.legalMoves)
    }
}

// Tests for ko.
extension BoardStateTests {
    func testStateBeforeKo() throws {
        let configuration = GameConfiguration(size: 4, komi: 0.1)
        var boardState = BoardState(gameConfiguration: configuration)
        // Create a board like this.
        // x/y 0 1 2 3
        //   0 . X O .
        //   1 X O . O
        //   2 . X O .
        //   3 . . . .
        boardState = try boardState.placingNewStone(at: Position(x: 0, y: 1))
        boardState = try boardState.placingNewStone(at: Position(x: 0, y: 2))
        boardState = try boardState.placingNewStone(at: Position(x: 1, y: 0))
        boardState = try boardState.placingNewStone(at: Position(x: 1, y: 1))
        boardState = try boardState.placingNewStone(at: Position(x: 2, y: 1))
        boardState = try boardState.placingNewStone(at: Position(x: 2, y: 2))
        boardState = boardState.passing()
        boardState = try boardState.placingNewStone(at: Position(x: 1, y: 3))

        XCTAssertEqual(.black, boardState.nextPlayerColor)
        XCTAssertNil(boardState.ko)
        XCTAssertEqual(8, boardState.playedMoveCount)
        XCTAssertEqual(7, boardState.stoneCount)
        let expectedLegalMoves = [
            Position(x: 0, y: 0),  // x: 0, y: 3 is a suiside for black.
            Position(x: 1, y: 2),
            Position(x: 2, y: 0),
            Position(x: 2, y: 3),
            Position(x: 3, y: 0),
            Position(x: 3, y: 1),
            Position(x: 3, y: 2),
            Position(x: 3, y: 3),
        ]
        XCTAssertEqual(expectedLegalMoves, boardState.legalMoves)
    }

    func testStateAfterKo() throws {
        let configuration = GameConfiguration(size: 4, komi: 0.1)
        var boardState = BoardState(gameConfiguration: configuration)
        // Before placing new stone.
        // x/y 0 1 2 3
        //   0 . X O .
        //   1 X O . O
        //   2 . X O .
        //   3 . . . .
        //
        // After placing new stone.
        // x/y 0 1 2 3
        //   0 . X O .
        //   1 X . X O    // white stone (x: 1, y: 1) is captured.
        //   2 . X O .
        //   3 . . . .
        boardState = try boardState.placingNewStone(at: Position(x: 0, y: 1))
        boardState = try boardState.placingNewStone(at: Position(x: 0, y: 2))
        boardState = try boardState.placingNewStone(at: Position(x: 1, y: 0))
        boardState = try boardState.placingNewStone(at: Position(x: 1, y: 1))
        boardState = try boardState.placingNewStone(at: Position(x: 2, y: 1))
        boardState = try boardState.placingNewStone(at: Position(x: 2, y: 2))
        boardState = boardState.passing()
        boardState = try boardState.placingNewStone(at: Position(x: 1, y: 3))
        boardState = try boardState.placingNewStone(at: Position(x: 1, y: 2))

        XCTAssertEqual(.white, boardState.nextPlayerColor)
        XCTAssertEqual(.some(Position(x: 1, y: 1)), boardState.ko)
        XCTAssertEqual(9, boardState.playedMoveCount)
        XCTAssertEqual(7, boardState.stoneCount)
        // x: 0, y: 0 is a suiside for white.
        // x: 1, y: 1 is a ko for white.
        let expectedLegalMoves = [
            Position(x: 0, y: 3),
            Position(x: 2, y: 0),
            Position(x: 2, y: 3),
            Position(x: 3, y: 0),
            Position(x: 3, y: 1),
            Position(x: 3, y: 2),
            Position(x: 3, y: 3),
        ]
        XCTAssertEqual(expectedLegalMoves, boardState.legalMoves)
    }
}

// Tests for score.
extension BoardStateTests {
    func testScore() throws {
        let configuration = GameConfiguration(size: 3, komi: 0.1)
        var boardState = BoardState(gameConfiguration: configuration)
        // Create a board like this.
        // x/y 0 1 2
        //   0 X X O
        //   1 . X O
        //   2 X O .
        boardState = try boardState.placingNewStone(at: Position(x: 0, y: 1))
        boardState = try boardState.placingNewStone(at: Position(x: 0, y: 2))
        boardState = try boardState.placingNewStone(at: Position(x: 1, y: 1))
        boardState = try boardState.placingNewStone(at: Position(x: 1, y: 2))
        boardState = try boardState.placingNewStone(at: Position(x: 2, y: 0))
        boardState = try boardState.placingNewStone(at: Position(x: 2, y: 1))
        boardState = try boardState.placingNewStone(at: Position(x: 0, y: 0))
        let areaForBlack: Float = 5.0
        let areaForWhite: Float = 4.0
        let scoreForBlack = areaForBlack - areaForWhite - configuration.komi
        XCTAssertEqual(scoreForBlack, boardState.score(for: .black))
        XCTAssertEqual(-scoreForBlack, boardState.score(for: .white))
    }

    func testScoreForDame() throws {
        let configuration = GameConfiguration(size: 3, komi: 0.1)
        var boardState = BoardState(gameConfiguration: configuration)
        // Create a board like this.
        // x/y 0 1 2
        //   0 X . O  // x: 0, y: 1 is a dame belonging to nobody.
        //   1 . X O
        //   2 X O .
        boardState = try boardState.placingNewStone(at: Position(x: 0, y: 0))
        boardState = try boardState.placingNewStone(at: Position(x: 0, y: 2))
        boardState = try boardState.placingNewStone(at: Position(x: 1, y: 1))
        boardState = try boardState.placingNewStone(at: Position(x: 1, y: 2))
        boardState = try boardState.placingNewStone(at: Position(x: 2, y: 0))
        boardState = try boardState.placingNewStone(at: Position(x: 2, y: 1))
        let areaForBlack: Float = 4.0
        let areaForWhite: Float = 4.0
        let scoreForBlack = areaForBlack - areaForWhite - configuration.komi
        XCTAssertEqual(scoreForBlack, boardState.score(for: .black))
        XCTAssertEqual(-scoreForBlack, boardState.score(for: .white))
    }
}

// Tests for equality.
extension BoardStateTests {
    func testEquality() throws {
        let configuration = GameConfiguration(size: 3, komi: 0.1)
        var boardState1 = BoardState(gameConfiguration: configuration)
        boardState1 = try boardState1.placingNewStone(at: Position(x: 0, y: 1))
        boardState1 = try boardState1.placingNewStone(at: Position(x: 0, y: 2))

        var boardState2 = BoardState(gameConfiguration: configuration)
        boardState2 = try boardState2.placingNewStone(at: Position(x: 0, y: 1))
        boardState2 = try boardState2.placingNewStone(at: Position(x: 0, y: 2))
        XCTAssertEqual(boardState1, boardState2)
    }

    func testInequalityForDifferentNextPlayers() throws {
        let configuration = GameConfiguration(size: 3, komi: 0.1)
        var boardState1 = BoardState(gameConfiguration: configuration)
        boardState1 = try boardState1.placingNewStone(at: Position(x: 0, y: 1))
        boardState1 = try boardState1.placingNewStone(at: Position(x: 0, y: 2))
        XCTAssertEqual(.black, boardState1.nextPlayerColor)

        var boardState2 = BoardState(gameConfiguration: configuration)
        boardState2 = try boardState2.placingNewStone(at: Position(x: 0, y: 1))
        boardState2 = try boardState2.placingNewStone(at: Position(x: 0, y: 2))
        boardState2 = boardState2.passing()  // passing flips the next player color.
        XCTAssertEqual(.white, boardState2.nextPlayerColor)
        XCTAssertNotEqual(boardState1, boardState2)
    }

    func testEqualityForEmptyBoards() {
        let configuration = GameConfiguration(size: 2, komi: 0.1)
        let emptyBoardState1 = BoardState(gameConfiguration: configuration)
        let emptyBoardState2 = BoardState(gameConfiguration: configuration)
        XCTAssertEqual(emptyBoardState1, emptyBoardState2)
    }

    func testEqualityForBoardWithKo() throws {
        XCTAssertEqual(try buildBoardWithKo(), try buildBoardWithKo())
    }

    private func buildBoardWithKo() throws -> BoardState{
        let configuration = GameConfiguration(size: 4, komi: 0.1)
        var boardState = BoardState(gameConfiguration: configuration)
        // Before placing new stone.
        // x/y 0 1 2 3
        //   0 . X O .
        //   1 X O . O
        //   2 . X O .
        //   3 . . . .
        //
        // After placing new stone.
        // x/y 0 1 2 3
        //   0 . X O .
        //   1 X . X O    // white stone (x: 1, y: 1) is captured.
        //   2 . X O .
        //   3 . . . .
        boardState = try boardState.placingNewStone(at: Position(x: 0, y: 1))
        boardState = try boardState.placingNewStone(at: Position(x: 0, y: 2))
        boardState = try boardState.placingNewStone(at: Position(x: 1, y: 0))
        boardState = try boardState.placingNewStone(at: Position(x: 1, y: 1))
        boardState = try boardState.placingNewStone(at: Position(x: 2, y: 1))
        boardState = try boardState.placingNewStone(at: Position(x: 2, y: 2))
        boardState = boardState.passing()
        boardState = try boardState.placingNewStone(at: Position(x: 1, y: 3))
        boardState = try boardState.placingNewStone(at: Position(x: 1, y: 2))
        XCTAssertEqual(.some(Position(x: 1, y: 1)), boardState.ko)
        return boardState
    }

    func testInequalityDueToKo() throws {
        let boardStateWithKo = try buildBoardWithKo()
        XCTAssertEqual(.some(Position(x: 1, y: 1)), boardStateWithKo.ko)

        // Passing twice to have the same board and same nextPlayerColor.
        let boardStateWithoutKo = boardStateWithKo.passing().passing()
        XCTAssertNil(boardStateWithoutKo.ko)

        XCTAssertNotEqual(boardStateWithKo, boardStateWithoutKo)
    }
}

extension BoardStateTests {
    static var allTests = [
        ("testEmptyBoard", testEmptyBoard),
        ("testPassing", testPassing),
        ("testPlacingNewStone", testPlacingNewStone),
        ("testStateBeforeKo", testStateBeforeKo),
        ("testStateAfterKo", testStateAfterKo),
        ("testScore", testScore),
        ("testScoreForDame", testScoreForDame),
        ("testEquality", testEquality),
        ("testEqualityForEmptyBoards", testEqualityForEmptyBoards),
        ("testInequalityForDifferentNextPlayers", testInequalityForDifferentNextPlayers),
        ("testEqualityForBoardWithKo", testEqualityForBoardWithKo),
        ("testInequalityDueToKo", testInequalityDueToKo),
    ]
}
