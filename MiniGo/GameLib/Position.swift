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

/// Represents a position in a Go game.
public struct Position: Hashable, Equatable {
    var x: Int
    var y: Int
}

/// Returns all valid neighbors for the given position on board.
extension Position {

    func neighbors(boardSize size: Int) -> [Position] {
        let neighbors = [
            Position(x: x+1, y: y),
            Position(x: x-1, y: y),
            Position(x: x, y: y+1),
            Position(x: x, y: y-1),
        ]

        return neighbors.filter { 0..<size ~= $0.x && 0..<size ~= $0.y }
    }
}
