// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

import ArgumentParser

struct WordSegCommand: ParsableCommand {
  static var configuration = CommandConfiguration(
    commandName: "WordSeg",
    abstract: """
      Runs training for the WordSeg model.
      """
  )

  @Flag(help: "Use eager backend (default).")
  var eager: Bool = false

  @Flag(help: "Use X10 backend.")
  var x10: Bool = false

  @Option(help: "Path to training data.")
  var trainingPath: String?

  @Option(help: "Path to validation data.")
  var validationPath: String?

  @Option(help: "Path to test data.")
  var testPath: String?

  @Option(help: "Maximum number of training epochs.")
  var maxEpochs: Int = 1000

  @Option(help: "Size of hidden layers.")
  var hiddenSize: Int = 512

  @Option(help: "Dropout rate.")
  var dropoutProbability: Double = 0.5

  @Option(help: "Power of the length penalty.")
  var order: Int = 5

  @Option(help: "Initial learning rate.")
  var learningRate: Float = 0.001

  @Option(help: "Weight of the length penalty.")
  var lambd: Float = 0.00075

  @Option(help: "Maximum length of a word.")
  var maxLength: Int = 10

  @Option(help: "Minimum frequency of a word.")
  var minFrequency: Int = 10 

  func validate() throws {
    guard !(eager && x10) else {
      throw ValidationError(
        "Can't specify both --eager and --x10 backends.")
    }
  }

  func run() throws {
    let backend: Backend = x10 ? .x10 : .eager

    let settings = WordSegSettings(
      trainingPath: trainingPath,
      validationPath: validationPath, testPath: testPath,
      hiddenSize: hiddenSize, dropoutProbability: dropoutProbability,
      order: order, lambd: lambd, maxEpochs: maxEpochs,
      learningRate: learningRate, backend: backend,
      maxLength: maxLength, minFrequency: minFrequency)

    try runTraining(settings: settings)
  }
}
