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

/// A policy asking the user to provide the next move.
public class HumanPolicy: Policy {

  public let participantName: String

  public init(participantName: String) {
      self.participantName = participantName
  }

  public func nextMove(for boardState: BoardState, after previousMove: Move?) -> Move {
    let legalMoves = boardState.legalMoves
    guard !legalMoves.isEmpty else {
      return .pass
    }

    func validator(_ position: Position) throws {
      guard legalMoves.contains(position) else {
        throw HumanInputError.invalidInput(message: "The move is not legal.")
      }
    }
    guard let position = promptAndReadMove(validatingWith: validator) else {
      return .pass
    }
    return .place(position: position)
  }
}

enum HumanInputError: Error {
  case emptyInput
  case invalidInput(message: String)
}

/// Gets the next move from user via stdio.
fileprivate func promptAndReadMove(validatingWith validator: (Position) throws -> ()) -> Position? {
  while true {
    do {
      print("Your input (x: -1, y: -1) means `pass`:")
      print("x: ", terminator: "")
      let x = try readCoordinate()
      print("y: ", terminator: "")
      let y = try readCoordinate()

      if x == -1 && y == -1 {
        return nil  // User chooses `pass`.
      }

      let position = Position(x: x, y: y)
      try validator(position)
      return position
    } catch let HumanInputError.invalidInput(message) {
      print("The input is invalid: \(message)")
      print("Please try again!")
    } catch HumanInputError.emptyInput {
      print("Empty input is now allowed.")
      print("Please try again!")
    } catch {
      print("Unknown error: \(error)")
      print("Please try again!")
    }
  }
}

fileprivate func readCoordinate() throws -> Int {
  guard let line = readLine() else {
    throw HumanInputError.emptyInput
  }
  guard let coordinate = Int(line) else {
    throw HumanInputError.invalidInput(message: "Coordinate must be Int.")
  }
  return coordinate
}
