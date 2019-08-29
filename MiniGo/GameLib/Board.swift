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

/// Holds the current board stones.
///
/// This struct allows caller to arbitrarily mutate the board information but
/// does not handle validation check for placing new stone. `BoardState` is
/// designed for that.
struct Board: Hashable {
    // Holds the stone `Color`s  for each position.
    private var stones: ShapedArray<Color?>
    let size: Int

    init(size: Int) {
        self.stones = ShapedArray<Color?>(repeating: nil, shape: [size, size])
        self.size = size
    }

    func color(at position: Position) -> Color? {
        assert(0..<size ~= position.x && 0..<size ~= position.y)
        return stones[position.x][position.y].scalars[0]
    }

    mutating func placeStone(at position: Position, withColor color: Color) {
        assert(0..<size ~= position.x && 0..<size ~= position.y)
        stones[position.x][position.y] = ShapedArraySlice(color)
    }

    mutating func removeStone(at position: Position) {
        assert(0..<size ~= position.x && 0..<size ~= position.y)
        stones[position.x][position.y] = ShapedArraySlice(nil)
    }
}

extension Board: CustomStringConvertible {
    var description: String {
        var output = ""

        // First, generates the head line, which looks like
        //
        //   x/y  0  1  2  3  4  5  6  7  8
        //
        // for a 9x9 board.
        output.append("\nx/y")

        // For board size <10, all numbers in head line are single digit. So, we only need one empty
        // space between them.
        //
        //   x/y 0 1 2 3 4 5 6 7 8
        //
        // For board size >=11, we need to print a space between two-digit numbers. So, spaces between
        // single-digit numbers are larger.
        //
        //   x/y  0  1  2  3  4  5  6  7  8  9 10 11
        for y in 0..<size {
            if size >= 11 {
                output.append(" ")
            }
            // As we cannot use Foundation, String(format:) method is not avaiable to use.
            if y < 10 {
                output.append(" \(y)")
            } else {
                output.append("\(y)")
            }
        }
        output.append("\n")

        // Similarly, we need two spaces between stones for size >= 11, but one space for small board.
        let gapBetweenStones = size <= 10 ? " " : "  "
        for x in 0..<size {
            // Prints row index.
            if x < 10 {
                output.append("  \(x)")  // Two leading spaces.
            } else {
                output.append(" \(x)")  // One leading space.
            }

            // Prints the color of stone at each position.
            for y in 0..<size {
                output.append(gapBetweenStones)
                guard let color = self.color(at: Position(x: x, y: y)) else {
                    // output.append("。")  // Empty position.
                    output.append("﹒")  // Empty position.
                    // output.append(".")  // Empty position.
                    continue
                }
                output.append(color == .black ? "🔵" : "⚪")
            }
            output.append("\n")
        }
        return output
    }
}
