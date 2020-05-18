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

import Datasets
import ModelSupport
import TensorFlow
import TextModels

// Model flags
let ndim = 512  // Hidden unit size.
// Training flags
let dropoutProb = 0.5  // Dropout rate.
let order = 5  // Power of length penalty.
let maxEpochs = 10  // Maximum number of training epochs.
let lambd: Float = 0.00075  // Weight of length penalty.
// Lexicon flags.
let maxLength = 10  // Maximum length of a string.
let minFreq = 10  // Minimum frequency of a string.

// Load user-provided data files.
let dataset: WordSegDataset
switch CommandLine.arguments.count {
case 1:
  dataset = try WordSegDataset()
case 2:
  dataset = try WordSegDataset(training: CommandLine.arguments[1])
case 3:
  dataset = try WordSegDataset(
    training: CommandLine.arguments[1], validation: CommandLine.arguments[2])
case 4:
  dataset = try WordSegDataset(
    training: CommandLine.arguments[1], validation: CommandLine.arguments[2],
    testing: CommandLine.arguments[3])
default:
  usage()
}

let lexicon = Lexicon(
  from: dataset.training,
  alphabet: dataset.alphabet,
  maxLength: maxLength,
  minFreq: minFreq
)

let modelParameters = SNLM.Parameters(
  ndim: ndim,
  dropoutProb: dropoutProb,
  chrVocab: dataset.alphabet,
  strVocab: lexicon,
  order: order
)

var model = SNLM(parameters: modelParameters)

let optimizer = Adam(for: model)

print("Starting training...")

for epoch in 1...maxEpochs {
  Context.local.learningPhase = .training
  var trainingLossSum: Float = 0
  var trainingBatchCount = 0
  for sentence in dataset.training {
    let (loss, gradients) = valueWithGradient(at: model) { model -> Float in
      let lattice = model.buildLattice(sentence, maxLen: maxLength)
      let score = lattice[sentence.count].semiringScore
      let expectedLength = exp(score.logr - score.logp)
      let loss = -1 * score.logp + lambd * expectedLength
      return loss
    }

    trainingLossSum += loss
    trainingBatchCount += 1
    optimizer.update(&model, along: gradients)

    if hasNaN(gradients) {
      print("Warning: grad has NaN")
    }
    if hasNaN(model) {
      print("Warning: model has NaN")
    }
  }

  guard let validationDataset = dataset.validation else {
    print(
      """
      [Epoch \(epoch)] \
      Training loss: \(trainingLossSum / Float(trainingBatchCount))
      """
    )
    continue
  }

  Context.local.learningPhase = .inference
  var validationLossSum: Float = 0
  var validationBatchCount = 0
  var validationCharacterCount = 0
  for sentence in validationDataset {
    let lattice = model.buildLattice(sentence, maxLen: maxLength)
    let score = lattice[sentence.count].semiringScore

    validationLossSum -= score.logp
    validationBatchCount += 1
    validationCharacterCount += sentence.count
  }

  let bpc = validationLossSum / Float(validationCharacterCount) / log(2)

  print(
    """
    [Epoch \(epoch)] \
    Bits per character: \(bpc) \
    Validation loss: \(validationLossSum / Float(validationBatchCount))
    """
  )
}

func hasNaN<T: KeyPathIterable>(_ t: T) -> Bool {
  for kp in t.recursivelyAllKeyPaths(to: Tensor<Float>.self) {
    if t[keyPath: kp].isNaN.any() { return true }
  }
  return false
}

func usage() -> Never {
  print(
    "\(CommandLine.arguments[0]) path/to/training_data.txt [path/to/validation_data.txt [path/to/test_data.txt]]"
  )
  exit(1)
}
