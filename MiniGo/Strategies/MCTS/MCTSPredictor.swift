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

public struct MCTSPrediction {
    /// The reward is for the next player for current board state.
    ///
    /// The reward has float value in range [-1, 1].
    let rewardForNextPlayer: Float

    struct Distribution {
        let positions: ShapedArray<Float>
        let pass: Float
    }

    /// The probability distribution over the position, including `pass`, to take for next move.
    ///
    /// The distribution is over all positions (including pass and illegal ones). It does not
    /// need to be normalized.
    let distribution: Distribution

    init(rewardForNextPlayer: Float, distribution: Distribution) {
        precondition(rewardForNextPlayer >= -1.0 && rewardForNextPlayer <= 1.0)
        self.rewardForNextPlayer = rewardForNextPlayer
        self.distribution = distribution
    }
}

/// Predicts the reward and distribution over positions for current `boardState`.
///
/// Predictor must be stateless and is expected to be used in multiple threads.
public protocol MCTSPredictor: class {
    func prediction(for boardState: BoardState) -> MCTSPrediction
}
