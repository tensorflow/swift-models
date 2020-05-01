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

enum IllegalMove: Error {
    case suicide
    case occupied

    /// A `ko` fight is a tactical and strategic phase that can arise in the game
    /// of go.
    ///
    /// The existence of ko fights is implied by the rule of ko, a special rule of
    /// the game that prevents immediate repetition of position, by a short 'loop'
    /// in which a single stone is captured, and another single stone immediately
    /// taken back.
    ///
    /// See https://en.wikipedia.org/wiki/Ko_fight for details.
    case ko
}

private enum PositionStatus: Equatable {
    case legal
    case illegal(reason: IllegalMove)
}

/// Represents an immutable snapshot of the current board state.
///
/// `BoardState` checks whether a new placed stone is legal or not. If so,
/// creates a new snapshot.
public struct BoardState {
    /// The game configuration.
    let gameConfiguration: GameConfiguration
    /// The color of the next player.
    let nextPlayerColor: Color

    /// The position of potential `ko`. See `IllegalMove.ko` for details.
    let ko: Position?

    /// All legal position to be considered as next move given the current board state.
    let legalMoves: [Position]

    /// All stones on the current board.
    let board: Board

    /// History of the previous board states (does not include current one).
    ///
    /// The most recent one is placed at index 0. The history count is truncated by
    /// `GameConfiguration.maxHistoryCount`.
    ///
    /// TODO(xiejw): Improve the efficient of history.
    let history: [Board]

    // General statistic.
    let playedMoveCount: Int
    let stoneCount: Int

    // Internal maintained states.
    private let libertyTracker: LibertyTracker

    /// Constructs an empty board state.
    init(gameConfiguration: GameConfiguration) {
        self.init(
            gameConfiguration: gameConfiguration,
            nextPlayerColor: .black,  // First player is always black.
            playedMoveCount: 0,
            stoneCount: 0,
            ko: nil,
            history: [],
            board: Board(size: gameConfiguration.size),
            libertyTracker: LibertyTracker(gameConfiguration: gameConfiguration)
        )
    }

    private init(
        gameConfiguration: GameConfiguration,
        nextPlayerColor: Color,
        playedMoveCount: Int,
        stoneCount: Int,
        ko: Position?,
        history: [Board],
        board: Board,
        libertyTracker: LibertyTracker
    ) {
        self.gameConfiguration = gameConfiguration
        self.nextPlayerColor = nextPlayerColor
        self.playedMoveCount = playedMoveCount
        self.stoneCount = stoneCount
        self.ko = ko

        assert(history.count <= gameConfiguration.maxHistoryCount)
        self.history = history

        self.libertyTracker = libertyTracker
        self.board = board
        precondition(board.size == gameConfiguration.size)

        if stoneCount == gameConfiguration.size * gameConfiguration.size {
            // Full board.
            self.legalMoves = []
        } else {
            self.legalMoves = board.allLegalMoves(
                ko: ko,
                libertyTracker: libertyTracker,
                nextPlayerColor: nextPlayerColor
            )
        }
    }

    /// Returns a new `BoardState` after current player passed.
    func passing() -> BoardState {
        var newHistory = self.history
        newHistory.insert(self.board, at: 0)
        if newHistory.count > gameConfiguration.maxHistoryCount {
            _ = newHistory.popLast()
        }
        return BoardState(
            gameConfiguration: self.gameConfiguration,
            nextPlayerColor: self.nextPlayerColor.opponentColor,
            playedMoveCount: self.playedMoveCount + 1,
            stoneCount: self.stoneCount,
            ko: nil,  // Reset ko.
            history: newHistory,
            board: self.board,
            libertyTracker: self.libertyTracker
        )
    }

    /// Returns a new `BoardState` after placing a new stone at `position`.
    func placingNewStone(at position: Position) throws -> BoardState {
        // Sanity Check first.
        if case .illegal(let reason) = board.positionStatus(
            at: position,
            ko: self.ko,
            libertyTracker: self.libertyTracker,
            nextPlayerColor: self.nextPlayerColor
        ) {
            throw reason
        }

        // Gets copies of libertyTracker and board. Updates both by placing new stone.
        let currentStoneColor = self.nextPlayerColor
        var newLibertyTracker = self.libertyTracker
        var newBoard = self.board

        // Makes attempt to guess the possible ko.
        let isPotentialKo = newBoard.isKoish(at: position, withNewStoneColor: currentStoneColor)

        // Updates libertyTracker and board by placing a new stone.
        let capturedStones = try newLibertyTracker.addStone(
            at: position, withColor: currentStoneColor)
        newBoard.placeStone(at: position, withColor: currentStoneColor)

        // Removes capturedStones
        for capturedStone in capturedStones {
            newBoard.removeStone(at: capturedStone)
        }

        // Updates stone count on board.
        let newStoneCount = self.stoneCount + 1 - capturedStones.count

        var newKo: Position?
        if let stone = capturedStones.first, capturedStones.count == 1, isPotentialKo {
            newKo = stone
        }

        var newHistory = self.history
        newHistory.insert(self.board, at: 0)
        if newHistory.count > gameConfiguration.maxHistoryCount {
            _ = newHistory.popLast()
        }

        return BoardState(
            gameConfiguration: self.gameConfiguration,
            nextPlayerColor: currentStoneColor == .black ? .white : .black,
            playedMoveCount: self.playedMoveCount + 1,
            stoneCount: newStoneCount,
            ko: newKo,
            history: newHistory,
            board: newBoard,
            libertyTracker: newLibertyTracker
        )
    }

    /// Returns the score of the player.
    func score(for playerColor: Color) -> Float {
        let scoreForBlackPlayer = self.board.scoreForBlackPlayer(komi: self.gameConfiguration.komi)
        switch playerColor {
        case .black:
            return scoreForBlackPlayer
        case .white:
            return -scoreForBlackPlayer
        }
    }
}

extension BoardState: CustomStringConvertible {
    public var description: String {
        return board.description
    }
}

extension BoardState: Equatable {
    public static func == (lhs: BoardState, rhs: BoardState) -> Bool {
        // The following line is the sufficient and necessary condition for "equal".
        return lhs.board == rhs.board && lhs.nextPlayerColor == rhs.nextPlayerColor
            && lhs.ko == rhs.ko && lhs.history == rhs.history
    }
}

extension Board {
    /// Calculates all legal moves on board.
    fileprivate func allLegalMoves(
        ko: Position?,
        libertyTracker: LibertyTracker,
        nextPlayerColor: Color
    ) -> [Position] {
        var legalMoves = [Position]()
        for x in 0..<self.size {
            for y in 0..<self.size {
                let position = Position(x: x, y: y)
                guard
                    .legal
                        == positionStatus(
                            at: position,
                            ko: ko,
                            libertyTracker: libertyTracker,
                            nextPlayerColor: nextPlayerColor
                        )
                else {
                    continue
                }

                legalMoves.append(position)
            }
        }
        return legalMoves
    }

    /// Checks whether a move is legal. If isLegal is false, reason will be set.
    fileprivate func positionStatus(
        at position: Position,
        ko: Position?,
        libertyTracker: LibertyTracker,
        nextPlayerColor: Color
    ) -> PositionStatus {
        guard self.color(at: position) == nil else { return .illegal(reason: .occupied) }
        guard position != ko else { return .illegal(reason: .ko) }

        guard
            !isSuicidal(
                at: position,
                libertyTracker: libertyTracker,
                nextPlayerColor: nextPlayerColor
            )
        else {
            return .illegal(reason: .suicide)
        }
        return .legal
    }

    /// A fast algorithm to check a possible suicidal move.
    ///
    /// This method assume the move is not `ko`.
    fileprivate func isSuicidal(
        at position: Position,
        libertyTracker: LibertyTracker,
        nextPlayerColor: Color
    ) -> Bool {
        var possibleLiberties = Set<Position>()

        for neighbor in position.neighbors(boardSize: self.size) {
            guard let group = libertyTracker.group(at: neighbor) else {
                // If the neighbor is not occupied, no liberty group, the position is OK.
                return false
            }
            if group.color == nextPlayerColor {
                possibleLiberties.formUnion(group.liberties)
            } else if group.liberties.count == 1 {
                // This move is capturing opponent's group. So, always legal.
                return false
            }
        }

        // After removing the new postion from liberties, if there is no liberties left, this move
        // is suicide.
        possibleLiberties.remove(position)
        return possibleLiberties.isEmpty
    }

    /// Checks whether the position is a potential ko, i.e., whether the position is surrounded by
    /// all sides belonging to the opponent.
    ///
    /// This is an approximated algorithm to find `ko`. See https://en.wikipedia.org/wiki/Ko_fight
    /// for details.
    fileprivate func isKoish(at position: Position, withNewStoneColor stoneColor: Color) -> Bool {
        precondition(self.color(at: position) == nil)
        let opponentColor = stoneColor.opponentColor
        let neighbors = position.neighbors(boardSize: self.size)
        return neighbors.allSatisfy { self.color(at: $0) == opponentColor }
    }
}

// Extends the Color (for player) to generate opponent's Color.
extension Color {
    fileprivate var opponentColor: Color {
        return self == .black ? .white : .black
    }
}

extension Board {
    /// Returns the score for black player.
    ///
    /// `komi` is the points added to the score of the player with the white stones as compensation
    /// for playing second.
    fileprivate func scoreForBlackPlayer(komi: Float) -> Float {
        // Makes a copy as we will modify it over time.
        var scoreBoard = self

        // First pass: Finds all empty positions on board.
        var emptyPositions = Set<Position>()
        for x in 0..<size {
            for y in 0..<size {
                let position = Position(x: x, y: y)
                if scoreBoard.color(at: position) == nil {
                    emptyPositions.insert(position)
                }
            }
        }

        // Second pass: Calculates the territory and borders for each empty position, if there is
        // any. If territory is surrounded by the stones in same color, fills that color in
        // territory.
        while !emptyPositions.isEmpty {
            let emptyPosition = emptyPositions.removeFirst()

            let (territory, borders) = territoryAndBorders(startingFrom: emptyPosition)
            guard !borders.isEmpty else {
                continue
            }

            // Fills the territory with black (or white) if the borders are all in black (or white).
            for color in Color.allCases {
                if borders.allSatisfy({ scoreBoard.color(at: $0) == color }) {
                    territory.forEach {
                        scoreBoard.placeStone(at: $0, withColor: color)
                        emptyPositions.remove($0)
                    }
                }
            }
        }

        // TODO(xiejw): Print out the modified board in debug mode.

        // Third pass: Counts stones now for scoring.
        var blackStoneCount = 0
        var whiteStoneCount = 0
        for x in 0..<size {
            for y in 0..<size {
                guard let color = scoreBoard.color(at: Position(x: x, y: y)) else {
                    // This board position does not belong to either player. Could be seki or dame.
                    // See https://en.wikipedia.org/wiki/Go_(game)#Seki_(mutual_life).
                    continue
                }
                switch color {
                case .black:
                    blackStoneCount += 1
                case .white:
                    whiteStoneCount += 1
                }
            }
        }
        return Float(blackStoneCount - whiteStoneCount) - komi
    }

    /// Finds the `territory`, all connected empty positions starting from `position`, and the
    /// `borders`, either black or white stones, surrounding the `territory`.
    ///
    /// The `position` must be an empty position. The returned `territory` contains empty positions
    /// only. The returned `borders` contains positions for placed stones. If the board is empty,
    /// `borders` will be empty.
    fileprivate func territoryAndBorders(
        startingFrom position: Position
    ) -> (territory: Set<Position>, borders: Set<Position>) {
        precondition(self.color(at: position) == nil)

        var territory = Set<Position>()
        var borders = Set<Position>()

        // Stores all candidates for the territory.
        var candidates: Set = [position]
        repeat {
            let currentPosition = candidates.removeFirst()
            territory.insert(currentPosition)

            for neighbor in currentPosition.neighbors(boardSize: self.size) {
                if self.color(at: neighbor) == nil {
                    if !territory.contains(neighbor) {
                        // We have not explored this (empty) position, so queue it up for
                        // processing.
                        candidates.insert(neighbor)
                    }
                } else {
                    // Insert the stone (either black or white) into borders.
                    borders.insert(neighbor)
                }
            }
        } while !candidates.isEmpty

        precondition(
            territory.allSatisfy { self.color(at: $0) == nil },
            "territory must be all empty (no stones).")
        precondition(
            borders.allSatisfy { self.color(at: $0) != nil },
            "borders cannot have empty positions.")
        return (territory, borders)
    }
}
