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

/// Represents an (immutable) configuration of a Go game.
public struct GameConfiguration {
    /// The board size of the game.
    let size: Int

    /// The points added to the score of the player with the white stones as compensation for playing
    /// second.
    let komi: Float

    /// The maximum number of board states to track.
    ///
    /// This does not include the the current board.
    let maxHistoryCount: Int

    /// If true, enables debugging information.
    let isVerboseDebuggingEnabled: Bool

    public init(
        size: Int,
        komi: Float,
        maxHistoryCount: Int = 7,
        isVerboseDebuggingEnabled: Bool = false
    ) {
        self.size = size
        self.komi = komi
        self.maxHistoryCount = maxHistoryCount
        self.isVerboseDebuggingEnabled = isVerboseDebuggingEnabled
    }
}

