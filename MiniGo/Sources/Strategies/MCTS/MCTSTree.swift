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
class MCTSTree {

    private let gameConfiguration: GameConfiguration
    private let predictor: MCTSPredictor

    var root: MCTSNode

    init(gameConfiguration: GameConfiguration, predictor: MCTSPredictor) {
        self.gameConfiguration = gameConfiguration
        self.predictor = predictor

        let emptyBoard = BoardState(gameConfiguration: gameConfiguration)

        let newNode = MCTSNode(
            gameConfiguration: gameConfiguration,
            predictor: predictor,
            boardState: emptyBoard)
        root = newNode
    }

    func promoteNewRoot(after previousMove: Move?) -> MCTSNode {
        guard let action = previousMove else {
            // Game just started. Returns the root node directly.
            assert(root.boardState.playedMoveCount == 0)
            return root
        }

        switch child(of: root, for: action) {
        case .existingNode(let node):
            root = node
        case .newNode(let node, _):
            root = node
        }
        return root
    }

    enum NodeKind {
        case existingNode(node: MCTSNode)
        case newNode(node: MCTSNode, rewardForNextPlayer: Float)
    }

    /// Returns the child node for `action`.
    func child(of node: MCTSNode, for action: Move) -> NodeKind {
        guard case let .stem(children) = node.kind else {
            fatalError("The node type is not expected.")
        }

        guard let actionNodePair = children.first(where: { $0.action == action }) else {
            fatalError("The action must exist for node \(node).")
        }

        let node = actionNodePair.child
        if node.statistic.totalVisitedCount == 0 {
            return .newNode(node: node, rewardForNextPlayer: node.statistic.rewardForNextPlayer)
        } else {
            return .existingNode(node: node)
        }
    }
}
