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

/// The MCTS tree used for one game playing.
final class MCTSTree {

    private let gameConfiguration: GameConfiguration
    private let predictor: MCTSPredictor

    var root: MCTSNode

    init(gameConfiguration: GameConfiguration, predictor: MCTSPredictor) {
        self.gameConfiguration = gameConfiguration
        self.predictor = predictor

        let emptyBoard = BoardState(gameConfiguration: gameConfiguration)
        let prediction = predictor.prediction(for: emptyBoard)

        let newNode = MCTSNode(
            boardSize: gameConfiguration.size,
            boardState: emptyBoard,
            distribution: prediction.distribution)
        root = newNode
    }

    func promoteNewRoot(after previousMove: Move?) -> MCTSNode {
        guard let action = previousMove else {
            // Game just started. Returns the root node directly.
            assert(root.boardState.playedMoveCount == 0)
            return root
        }

        // Tries to find the new root if it is already one of the children of current root.
        if let newRoot = root.children[action] {
            root = newRoot
            return newRoot
        }

        // Creates a new node and promotes it.
        let newBoardState: BoardState
        switch action {
        case .pass:
            newBoardState = root.boardState.passing()
        case .place(let position):
            do {
                newBoardState = try root.boardState.placingNewStone(at: position)
            } catch {
                fatalError("MCTS algorithm should never emit an illegal action. Got error: \(error).")
            }
        }

        // creates the new node by calling the predictor.
        let prediction = predictor.prediction(for: newBoardState)

        root = MCTSNode(
            boardSize: gameConfiguration.size,
            boardState: newBoardState,
            distribution: prediction.distribution)
        return root
    }

    enum NodeKind {
        case existingNode(node: MCTSNode)
        case newNode(node: MCTSNode, rewardForNextPlayer: Float)
    }

    /// Returns the child node for `action`.
    func child(of node: MCTSNode, for action: Move) -> NodeKind {
        if let child = node.children[action] {
            return .existingNode(node: child)
        }

        let newBoardState: BoardState
        switch action {
        case .pass:
            newBoardState = node.boardState.passing()
        case .place(let position):
            do {
                newBoardState = try node.boardState.placingNewStone(at: position)
            } catch {
                fatalError("MCTS algorithm should never emit an illegal action. Got error: \(error).")
            }
        }

        // Creates the new node by calling the predictor.
        let prediction = predictor.prediction(for: newBoardState)

        // TODO(xiejw): Implement noise injection for predictions.
        let newNode = MCTSNode(
            boardSize: gameConfiguration.size,
            boardState: newBoardState,
            distribution: prediction.distribution)
        node.children[action] = newNode
        return .newNode(node: newNode, rewardForNextPlayer: prediction.rewardForNextPlayer)
    }
}
