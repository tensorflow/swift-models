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

/// The Monte Carlo tree search (MCTS) algorithrm based policy.
public class MCTSPolicy: Policy {
    public let participantName: String

    private let configuration: MCTSConfiguration
    private var tree: MCTSTree

    public init(participantName: String, predictor: MCTSPredictor, configuration: MCTSConfiguration)
    {
        self.participantName = participantName
        self.configuration = configuration
        self.tree = MCTSTree(
            gameConfiguration: configuration.gameConfiguration, predictor: predictor)
    }

    public func nextMove(for boardState: BoardState, after previousMove: Move?) -> Move {
        // Stage 1: Promote the corresponding node in the tree to become the new root. This purges any
        // old nodes which are not used anymore.
        let root = tree.promoteNewRoot(after: previousMove)
        assert(
            root.boardState == boardState,
            "Expected board \(boardState),\n got: \(root.boardState).")

        // Stage 2: Runs simulations to expand the tree.
        for _ in 0..<configuration.simulationCountForOneMove {
            runOneSimulation(previousMove: previousMove)
        }

        // Stage 3: Select a move.
        var exploreMove = false
        if case .exploreMovesInEarlyStage(let maximumMoveCount) = configuration.explorationOption,
            boardState.playedMoveCount <= maximumMoveCount
        {
            exploreMove = true
        }

        // Stage 4: Promotes again to future trim the tree.
        let move = root.nextMove(withExplorationEnabled: exploreMove)
        _ = tree.promoteNewRoot(after: move)
        return move
    }

    private func runOneSimulation(previousMove: Move?) {
        var consecutivePassCount = 0
        if case .pass? = previousMove {
            consecutivePassCount = 1
        }

        var rewardForBlackPlayer: Float?
        var visitedActions: [Move] = []
        var visitedMCTSNodes: [MCTSNode] = []

        var currentNodeKind: MCTSTree.NodeKind = .existingNode(node: tree.root)

        // Expands the tree until the current game is finished, a new node is seen, or the maximum game
        // depth is reached.
        explandTree: while true {
            assert(consecutivePassCount <= 1)

            switch currentNodeKind {
            case let .newNode(node, rewardForNextPlayer):
                // Reaches a new node: Expand this node and return.
                var reward = rewardForNextPlayer
                if node.boardState.nextPlayerColor == .white {
                    reward *= -1.0
                }
                rewardForBlackPlayer = reward
                break explandTree

            case .existingNode(let node):
                let action = node.actionByPUCT
                visitedActions.append(action)
                visitedMCTSNodes.append(node)

                // To avoid infinite tree expansion, quit expanding once max game depth is reached.
                if node.boardState.playedMoveCount >= configuration.maxGameDepth {
                    rewardForBlackPlayer = node.boardState.rewardForBlackPlayer()
                    break explandTree
                }

                switch action {
                case .pass:
                    consecutivePassCount += 1

                    // Two consecutive passes end the game. Quit expanding at this point.
                    if consecutivePassCount == 2 {
                        rewardForBlackPlayer = node.boardState.rewardForBlackPlayer()
                        break explandTree
                    }

                case .place(_):
                    consecutivePassCount = 0
                }

                currentNodeKind = tree.child(of: node, for: action)
            }
        }

        // Backup the reward to all visited notes.
        assert(visitedActions.count == visitedMCTSNodes.count)
        guard let finalRewardForBlackPlayer = rewardForBlackPlayer else {
            fatalError(
                "The reward must be set during simulation before quiting the tree expanding.")
        }
        for (index, node) in visitedMCTSNodes.enumerated() {
            node.backUp(
                for: visitedActions[index],
                withRewardForBlackPlayer: finalRewardForBlackPlayer)
        }
    }
}

extension BoardState {
    /// Converts the score to reward for black player.
    ///
    /// Score is a numerical value according to the scoring rule, like area score. Reward takes
    /// binary values, 1 for win and -1 for lose.
    fileprivate func rewardForBlackPlayer() -> Float {
        let scoreForBlackPlayer = score(for: .black)
        switch scoreForBlackPlayer.sign {
        case .plus: return 1.0
        case .minus: return -1.0
        }
    }
}
