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

/// Tree node for the MCTS algorithm.
class MCTSNode {

    enum Kind {
        case stem([(action: Move, child: MCTSNode)])
    }

    // To avoid the tree grows automatically during construction, uses lazy var here.
    lazy var kind: Kind = {
        var actions: [Move] = [.pass]  // .pass must be the first one.
        actions.reserveCapacity(boardState.legalMoves.count + 1)
        boardState.legalMoves.forEach {
            actions.append(.place(position: $0))
        }

        let children = actions.map { action -> (action: Move, child: MCTSNode) in
            let newBoardState: BoardState
            switch action {
            case .pass:
                newBoardState = boardState.passing()
            case .place(let position):
                do {
                    newBoardState = try boardState.placingNewStone(at: position)
                } catch {
                    fatalError("MCTS algorithm should never emit an illegal action. Got error: \(error).")
                }
            }

            let newNode = MCTSNode(
                gameConfiguration: gameConfiguration,
                predictor: predictor,
                boardState: newBoardState
            )
            return (action, newNode)
        }
        return .stem(children)
    }()

    lazy var statistic: MultiArmedBanditStatistic = {
        // Creates the new node by calling the predictor.
        let prediction = predictor.prediction(for: boardState)

        // TODO(xiejw): Implement noise injection for predictions.
        return MultiArmedBanditStatistic(
            boardSize: gameConfiguration.size,
            boardState: boardState,
            distribution: prediction.distribution,
            rewardForNextPlayer: prediction.rewardForNextPlayer)
    }()

    private let gameConfiguration: GameConfiguration
    private let predictor: MCTSPredictor
    let boardState: BoardState

    init(gameConfiguration: GameConfiguration,
         predictor: MCTSPredictor,
         boardState: BoardState
    ) {
        self.gameConfiguration = gameConfiguration
        self.predictor = predictor
        self.boardState = boardState
    }
}

class MultiArmedBanditStatistic {

    private let boardSize: Int

    /// Total visited count for this node during simulations.
    var totalVisitedCount: Int = 0

    /// Rewards for next player.
    let rewardForNextPlayer: Float

    /// The corresponding board state for this node.
    let boardState: BoardState

    struct Action {
        let move: Move
        var prior: Float
        var qValueTotal: Float
        var visitedCount: Int
    }

    /// The `actionSpace` consists of all legal actions for current `BoardState`. The first action is
    /// `.pass`, followed by all legal positions.
    ///
    /// Note: `prior` in `actionSpace` must be normalized to form a valid probability.
    var actionSpace: [Action]

    /// Creates a MCTS node.
    ///
    /// - Precondition: The `distribution` is not expected to be normalized. And it is allowed to
    /// have positive values for illegal positions.
    init(
        boardSize: Int,
        boardState: BoardState,
        distribution: MCTSPrediction.Distribution,
        rewardForNextPlayer: Float
    ) {
        self.boardSize = boardSize
        self.boardState = boardState
        self.rewardForNextPlayer = rewardForNextPlayer

        var actions: [Move] = [.pass]  // .pass must be the first one.

        actions.reserveCapacity(boardState.legalMoves.count + 1)
        boardState.legalMoves.forEach {
            actions.append(.place(position: $0))
        }

        var priorOverActions = Array(repeating: Float(0), count: actions.count)
        var sum: Float = 0
        for (index, action) in actions.enumerated() {
            let prior: Float
            switch action {
            case .pass:
                assert(index == 0)
                prior = distribution.pass
            case .place(let position):
                assert(index > 0)
                prior = distribution.positions[position.x][position.y].scalars[0]
            }
            sum += prior
            priorOverActions[index] = prior
        }

        self.actionSpace = actions.enumerated().map {
            Action(move: $1, prior: priorOverActions[$0] / sum, qValueTotal: 0.0, visitedCount: 0)
        }
    }
}

/// Supports the node backing up.
extension MCTSNode {
    /// Backs up the reward.
    func backUp(for move: Move, withRewardForBlackPlayer rewardForBlackPlayer: Float) {
        guard let index = statistic.actionSpace.firstIndex(where: { $0.move == move }) else {
            fatalError(
                "The action \(move) taken must be legal (all legal actions: \(statistic.actionSpace)).")
        }

        statistic.totalVisitedCount += 1
        statistic.actionSpace[index].visitedCount += 1
        statistic.actionSpace[index].qValueTotal += rewardForBlackPlayer *
            (boardState.nextPlayerColor == .black ? 1.0 : -1.0)
    }
}

/// Supports selecting the action.
extension MCTSNode {
    /// Returns the next move to take based on current learned statistic in Node.
    func nextMove(withExplorationEnabled: Bool) -> Move {
        precondition(
            statistic.totalVisitedCount > 0,
            "The node has not been visited after creation.")
        if withExplorationEnabled {
            return sampleFromPMF(statistic.actionSpace) { Float($0.visitedCount) }.move
        } else {
            return maxScoringElement(statistic.actionSpace) { Float($0.visitedCount) }.move
        }
    }

    /// Selects the action based on PUCT algorithm for simulation.
    ///
    /// PUCT stands for predictor + UCT, where UCT stands for UCB applied to trees. The
    /// action is selected based on the statistic in the search tree and has some levels of
    /// exploration. Initially, this algorithm prefers action with high prior probability and low
    /// visit count but asymptotically prefers action with high action value.
    ///
    /// See the AlphaGoZero paper and its references for details.
    var actionByPUCT: Move {
        guard statistic.totalVisitedCount > 0 else {
            // If the node has not be visited after creation, we select the move based on prior
            // probability.
            return nextMoveWithHighestPrior
        }
        return nextMoveWithHighestActionValue
    }
}

extension MCTSNode {
    private var nextMoveWithHighestPrior: Move {
        return maxScoringElement(statistic.actionSpace) { $0.prior }.move
    }

    private var nextMoveWithHighestActionValue: Move {
        return maxScoringElement(
            statistic.actionSpace,
            withScoringFunction: {
                // See the AlphaGoZero paper ("Methods" -> "Select" section) for the formula of action
                // value.
                let visitedCount = $0.visitedCount

                var actionValue = $0.prior *
                    (Float(statistic.totalVisitedCount) / (1.0 + Float(visitedCount))).squareRoot()

                if visitedCount > 0 {
                    actionValue += $0.qValueTotal / Float(visitedCount)
                }
                return actionValue
            }
        ).move
    }
}

extension MCTSNode {
    /// A general algorithm to find the element with highest score. If there are multiple,
    /// breaks the tie randomly.
    private func maxScoringElement<T>(
        _ elements: [T],
        withScoringFunction scoringFunction: (T) -> Float
    ) -> T {
        precondition(elements.count > 0)
        var candidateIndexes = [0]
        var highestValue = scoringFunction(elements[0])

        for index in 1..<elements.count {
            let v = scoringFunction(elements[index])
            if v > highestValue {
                highestValue = v
                candidateIndexes.removeAll()
                candidateIndexes.append(index)
            } else if abs(v - highestValue) < .ulpOfOne {
                precondition(candidateIndexes.count > 0)
                candidateIndexes.append(index)
            }
        }

        let candidateCount = candidateIndexes.count
        assert(candidateCount > 0)

        // Breaks the tie randomly.
        let candidateIndex = candidateIndexes[Int.random(in: 0..<candidateCount)]
        return elements[candidateIndex]
    }

    /// Samples an element according to the PMF.
    private func sampleFromPMF<T>(_ elements: [T], with pmfFunction: (T) -> Float) -> T {
        precondition(elements.count > 0)
        var cdf: [Float] = []
        var currentSum: Float = 0.0
        for element in elements {
            let probability = pmfFunction(element)
            assert(probability >= 0)
            currentSum += probability
            cdf.append(currentSum)
        }

        let sampleSpace = 10000
        let sample = Int.random(in: 0..<sampleSpace)
        let threshold = Float(sample) / Float(sampleSpace) * currentSum

        for (i, element) in elements.enumerated() where threshold < cdf[i] {
            return element
        }
        return elements[elements.count - 1]
    }
}
