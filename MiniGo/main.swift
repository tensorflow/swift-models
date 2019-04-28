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

import Foundation
import TensorFlow
import MiniGo

let boardSize = 19
let simulationCountForOneMove = 40

let gameConfiguration = GameConfiguration(
    size: boardSize,
    komi: 7.5,
    isVerboseDebuggingEnabled: false)

// Creates the GoModel and loads the checkpoint.
print("Loading checkpoint into GoModel. Might take a while.")
let modelConfig = ModelConfiguration(boardSize: boardSize)
var model = GoModel(configuration: modelConfig)

guard FileManager.default.fileExists(atPath: "./MiniGoCheckpoint/000939-heron.data-00000-of-00001")
    else { fatalError("Please download the MiniGo checkpoint according to the README.md.") }
let reader = PythonCheckpointReader(path: "./MiniGoCheckpoint/000939-heron")
model.load(from: reader)

let predictor = MCTSModelBasedPredictor(boardSize: boardSize, model: model)

// Pick up policies to play. The first policy in `participants` plays black.
// Current available policies are:
//   - RandomPolicy
//   - HumanPolicy
//   - MCTSPolicy
let mctsConfiguration = MCTSConfiguration(
    gameConfiguration: gameConfiguration,
    simulationCountForOneMove: simulationCountForOneMove)

try playOneGame(
    gameConfiguration: gameConfiguration,
    participants: [
        MCTSPolicy(participantName: "black", predictor: predictor,
                   configuration: mctsConfiguration),
        MCTSPolicy(participantName: "white", predictor: predictor,
                   configuration: mctsConfiguration),
    ])

