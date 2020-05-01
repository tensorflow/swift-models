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

/// The configuration for MCTS algorithm.
public struct MCTSConfiguration {
    /// The configuration of the Go game.
    let gameConfiguration: GameConfiguration

    /// The total number of simulations to run for each move.
    let simulationCountForOneMove: Int

    /// The maximum game depth to expand the tree during simulation.
    ///
    /// The maximum game depth is compared with `BoardState.playedMoveCount`. Once reached, score the
    /// board immediately. This is used to avoid infinite game plays during simulation.
    let maxGameDepth: Int

    public enum ExplorationOption {
        /// Disable exploration in MCTS algorithm when selecting a move.
        ///
        /// This should be used for real game play to generate strongest move.
        case noExploration

        /// Enable exploration in early stage of the game.
        ///
        /// To be specific, if the `BoardState.playedMoveCount` is no larger than the
        /// `maximumMoveCountToExplore`, enable the / exploration to select move. This helps improving
        /// the early stage diversity.
        ///
        /// This is recommended for self plays to generate training data.
        case exploreMovesInEarlyStage(maximumMoveCountToExplore: Int)
    }

    /// The configuration for exploration.
    let explorationOption: ExplorationOption

    // The default value, 1600, for `simulationCountForOneMove` was the number used by the AlphaGoZero
    // paper.
    //
    // If `maxGameDepth` is `nil`, it will be set to (gameConfiguration.size)^2 * 1.4 according to the
    // MiniGo reference model, i.e., 505 moves for 19x19, 113 for 9x9. The AlphaGo paper chooses 2.0
    // instead of 1.4.
    public init(
        gameConfiguration: GameConfiguration,
        simulationCountForOneMove: Int = 1600,
        maxGameDepth: Int? = nil,
        explorationOption: ExplorationOption = .noExploration
    ) {
        self.gameConfiguration = gameConfiguration

        precondition(simulationCountForOneMove > 0)
        self.simulationCountForOneMove = simulationCountForOneMove

        let maxGameDepthValue =
            maxGameDepth ?? Int(Float(gameConfiguration.size * gameConfiguration.size) * 1.4)
        precondition(maxGameDepthValue > 0)
        self.maxGameDepth = maxGameDepthValue

        self.explorationOption = explorationOption
    }
}
