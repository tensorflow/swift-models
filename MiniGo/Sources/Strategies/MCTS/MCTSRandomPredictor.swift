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

import TensorFlow

/// A random `MCTSPredictor` predicting the next move and reward with random numbers.
///
/// This is mainly for testing and debugging purposes.
public class MCTSRandomPredictor: MCTSPredictor {
    private let boardSize: Int

    public init(boardSize: Int) {
        self.boardSize = boardSize
    }

    public func prediction(for boardState: BoardState) -> MCTSPrediction {
        let distribution = MCTSPrediction.Distribution(
            positions: ShapedArray<Float>(shape: [boardSize, boardSize], repeating: 1.0),
            pass: 1.0)

        // Randomize the reward (range: 0.0 +/- 0.05).
        let reward = 0.0 + (Float(Int.random(in: 0..<100)) - 50.0) / 1000.0
        return MCTSPrediction(rewardForNextPlayer: reward, distribution: distribution)
    }
}
