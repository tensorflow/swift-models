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

public protocol InferenceModel {
  /// Predicts the model output based on input tensor.
  func prediction(input: Tensor<Float>) -> GoModelOutput
}

/// A ResNet-like model based `MCTSPredictor` predicting the next move and reward.
public class MCTSModelBasedPredictor: MCTSPredictor {
  // Maintainer Note: As of Feb 2019, `Tensor` shape related APIs require `Int32`, but
  // `ShapedArray` requires `Int` for shape. So, we have to pick one here. Consider the fact we will
  // use `Tensor` anyway in future but might change another implementation to replace `ShapedArray`,
  // `Int32` is chosen here.
  private let boardSize: Int32
  private var model: InferenceModel

  public init(boardSize: Int, model: InferenceModel) {
    guard boardSize == 19 else {
      fatalError("GoModel only supports boardSize=19 for now.")
    }
    self.boardSize = Int32(boardSize)
    self.model = model
  }

  public func prediction(for boardState: BoardState) -> MCTSPrediction {
    let modelInput = boardState.featurePlanes()
    let inference = model.prediction(input: modelInput)

    let policy = inference.policy.flattened()
    assert(policy.shape == [boardSize*boardSize + 1])

    // The first boardSize * boardSize elements are placed in `positions`.
    // The final value is for `pass`.
    let distribution = MCTSPrediction.Distribution(
      positions: policy[0..<boardSize*boardSize].reshaped(to: [boardSize, boardSize]).array,
      pass: policy[policy.scalarCount - 1].scalarized())

    assert(inference.value.shape == [1])
    var reward = inference.value.scalarized()

    // We occasionally see the model output falls out of the expected range, which should never
    // happen given the final activation funciton is `tanh`.
    //
    // To avoid crash, we log the case here and do value clipping.
    if reward > 1.0 {
      print("Reward is out of range: value \(reward). \n \(boardState)")
      reward = 1.0
    }
    if reward < -1.0 {
      print("Reward is out of range: value \(reward). \n \(boardState)")
      reward = -1.0
    }
    return MCTSPrediction(rewardForNextPlayer: reward, distribution: distribution)
  }
}

extension BoardState {

  /// Returns the feature planes as `Tensor`.
  ///
  /// The output `Tensor` has shape `[1, boardSize, boardSize, 17]`.
  ///
  /// For reference, see the AlphaGo Zero paper, Section: Method -> Neural network architecture.
  fileprivate func featurePlanes() -> Tensor<Float> {
    assert(gameConfiguration.maxHistoryCount <= 7, "Only support at most 8 board states in total.")

    let boardSize = gameConfiguration.size

    var featurePlanes = ShapedArray<Float>(shape: [17, boardSize, boardSize], repeating: 0.0)

    // The first 16 feature planes represent recent board states. Each board state needs two planes.
    //
    // First, sets the feature planes for the current board state.
    var featurePlanesForOldestBoard = self.board.binaryFeaturePlanes(
      activePlayerColor: self.nextPlayerColor)
    featurePlanes[0...1] = featurePlanesForOldestBoard

    // Then, sets the feature planes for each board state in history.
    var nextIndex = 2
    for boardInHistory in self.history {
      featurePlanesForOldestBoard = boardInHistory.binaryFeaturePlanes(
        activePlayerColor: self.nextPlayerColor)
      featurePlanes[nextIndex...nextIndex+1] = featurePlanesForOldestBoard
      nextIndex += 2
    }

    // Finally, sets the remaining feature planes as the one for the last one.
    //
    // AlphaGo sets the remaining feature planes as all zeros (see "Method" -> "Neural network
    // architecture" section). But MiniGo reference model repeats the oldest board. We followed the
    // latter here.
    assert(nextIndex == (self.history.count + 1) * 2)
    while nextIndex < 16 {
      featurePlanes[nextIndex...(nextIndex+1)] = featurePlanesForOldestBoard
      nextIndex += 2
    }

    // The final feature plane represents the color to play. 1.0 if black is to play or 0.0 if white
    // is to play.
    featurePlanes[16] = ShapedArraySlice<Float>(
      shape: [boardSize, boardSize],
      repeating: self.nextPlayerColor == .black ? 1.0 : 0.0)

    let featureTensor = Tensor(featurePlanes)

    // The Go prediction network expects the input tensor to be in `[batch, boardSize, boardSize,
    // featurePlanes]` order.
    //
    // Rotate our inputs to this order by transposing and reshape to a single-element batch.
    return featureTensor.transposed(
      withPermutations: 1, 2, 0
    ).reshaped(to: [1, Int32(boardSize), Int32(boardSize), 17])
  }
}

extension Board {
  /// Converts the board state to binary feature planes.
  ///
  /// The first plane has 1 on position where the stone's color matches `activePlayerColor`.
  /// The second plane has 1 on position where the stone's color matches opponent's color.
  ///
  /// `activePlayerColor` is not same as `BoardState.nextPlayerColor`. The active player color (the
  /// player to play) being filled depends on the point in history we are filling.
  ///
  /// Consider a board state A, white is playing and the state history is [A, B, C, D].
  ///
  /// Calls to binaryFeaturePlanes will be:
  ///
  ///     binaryFeaturePlanes(state: A, activePlayerColor: .white)
  ///     binaryFeaturePlanes(state: B, activePlayerColor: .white)
  ///     binaryFeaturePlanes(state: C, activePlayerColor: .white)
  ///     binaryFeaturePlanes(state: D, activePlayerColor: .white)
  fileprivate func binaryFeaturePlanes(activePlayerColor: Color) -> ShapedArraySlice<Float> {
    let boardSize = self.size

    let opponentColor: Color = activePlayerColor == .black ? .white : .black

    var result = ShapedArraySlice<Float>(shape: [2, boardSize, boardSize], repeating: 0.0)
    for x in 0..<boardSize {
      for y in 0..<boardSize {
        guard let stoneColor = color(at: Position(x: x, y: y)) else {
          continue
        }

        if stoneColor == activePlayerColor {
          result[0][x][y] = ShapedArraySlice(1.0)
        } else {
          assert(stoneColor == opponentColor)
          result[1][x][y] = ShapedArraySlice(1.0)
        }
      }
    }
    return result
  }
}
