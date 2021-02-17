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

#if os(Windows)
#if canImport(CRT)
import CRT
#else
import MSVCRT
#endif
#endif

internal func runTraining(settings: WordSegSettings) throws {
  var trainingLossHistory = [Float]()  // Keep track of loss.
  var validationLossHistory = [Float]()  // Keep track of loss.
  var noImprovements = 0  // Consecutive epochs without improvements to loss.

  // Load user-provided data files.
  let dataset: WordSegDataset
  if settings.trainingPath == nil {
    dataset = try WordSegDataset()
  } else {
    dataset = try WordSegDataset(
      training: settings.trainingPath!, validation: settings.validationPath,
      testing: settings.testPath)
  }

  let sequences = dataset.trainingPhrases.map { $0.numericalizedText }
  let lexicon = Lexicon(
    from: sequences,
    alphabet: dataset.alphabet,
    maxLength: settings.maxLength,
    minFrequency: settings.minFrequency
  )

  let modelParameters = SNLM.Parameters(
    hiddenSize: settings.hiddenSize,
    dropoutProbability: Double(settings.dropoutProbability),
    alphabet: dataset.alphabet,
    lexicon: lexicon,
    order: settings.order
  )

  let device: Device
  switch settings.backend {
  case .eager:
    device = Device.defaultTFEager
  case .x10:
    device = Device.defaultXLA
  }

  var model = SNLM(parameters: modelParameters)
  model.move(to: device)

  var optimizer = Adam(for: model, learningRate: settings.learningRate)
  optimizer = Adam(copying: optimizer, to: device)

  print("Starting training...")

  for epoch in 1...settings.maxEpochs {
    Context.local.learningPhase = .training
    var trainingLossSum: Float = 0
    var trainingBatchCount = 0
    let trainingBatchCountTotal = dataset.trainingPhrases.count
    for phrase in dataset.trainingPhrases {
      let sentence = phrase.numericalizedText
      let (loss, gradients) = valueWithGradient(at: model) { model -> Tensor<Float> in
        let lattice = model.buildLattice(sentence, maxLen: settings.maxLength, device: device)
        let score = lattice[sentence.count].semiringScore
        let expectedLength = exp(score.logr - score.logp)
        let loss = -1 * score.logp + settings.lambd * expectedLength
        return Tensor(loss, on: device)
      }

      let lossScalarized = loss.scalarized()
      if trainingBatchCount % 10 == 0 {
        let bpc = getBpc(loss: lossScalarized, characterCount: sentence.count)
        print(
          """
          [Epoch \(epoch)] (\(trainingBatchCount)/\(trainingBatchCountTotal)) | Bits per character: \(bpc)
          """
        )
      }

      trainingLossSum += lossScalarized
      trainingBatchCount += 1

      optimizer.update(&model, along: gradients)
      LazyTensorBarrier()
      if hasNaN(gradients) {
        print("Warning: grad has NaN")
      }
      if hasNaN(model) {
        print("Warning: model has NaN")
      }
    }

    // Decrease the learning rate if loss is stagnant.
    let trainingLoss = trainingLossSum / Float(trainingBatchCount)
    trainingLossHistory.append(trainingLoss)
    reduceLROnPlateau(lossHistory: trainingLossHistory, optimizer: optimizer)

    if dataset.validationPhrases.count < 1 {
      print(
        """
        [Epoch \(epoch)] \
        Training loss: \(trainingLoss)
        """
      )

      // Stop training when loss stops improving.
      if terminateTraining(
        lossHistory: trainingLossHistory,
        noImprovements: &noImprovements)
      {
        break
      }

      continue
    }

    Context.local.learningPhase = .inference
    var validationLossSum: Float = 0
    var validationBatchCount = 0
    var validationCharacterCount = 0
    var validationPlainText: String = ""
    for phrase in dataset.validationPhrases {
      let sentence = phrase.numericalizedText
      var lattice = model.buildLattice(sentence, maxLen: settings.maxLength, device: device)
      let score = lattice[sentence.count].semiringScore

      validationLossSum -= score.logp
      validationBatchCount += 1
      validationCharacterCount += sentence.count

      // View a sample segmentation once per epoch.
      if validationBatchCount == dataset.validationPhrases.count {
        let bestPath = lattice.viterbi(sentence: phrase.numericalizedText)
        validationPlainText = Lattice.pathToPlainText(path: bestPath, alphabet: dataset.alphabet)
      }
    }

    let bpc = getBpc(loss: validationLossSum, characterCount: validationCharacterCount)
    let validationLoss = validationLossSum / Float(validationBatchCount)

    print(
      """
      [Epoch \(epoch)] Learning rate: \(optimizer.learningRate)
        Validation loss: \(validationLoss), Bits per character: \(bpc)
        \(validationPlainText)
      """
    )

    // Stop training when loss stops improving.
    validationLossHistory.append(validationLoss)
    if terminateTraining(lossHistory: validationLossHistory, noImprovements: &noImprovements) {
      break
    }
  }
}

fileprivate func getBpc(loss: Float, characterCount: Int) -> Float {
  return loss / Float(characterCount) / log(2)
}

fileprivate func hasNaN<T: KeyPathIterable>(_ t: T) -> Bool {
  for kp in t.recursivelyAllKeyPaths(to: Tensor<Float>.self) {
    if t[keyPath: kp].isNaN.any() { return true }
  }
  return false
}

fileprivate func terminateTraining(
  lossHistory: [Float], noImprovements: inout Int, patience: Int = 5
) -> Bool {
  if lossHistory.count <= patience { return false }
  let window = Array(lossHistory.suffix(patience))
  guard let loss = lossHistory.last else { return false }

  if window.min() == loss {
    if window.max() == loss { return true }
    noImprovements = 0
  } else {
    noImprovements += 1
    if noImprovements >= patience { return true }
  }

  return false
}

fileprivate func reduceLROnPlateau(
  lossHistory: [Float], optimizer: Adam<SNLM>,
  factor: Float = 0.25
) {
  let threshold: Float = 1e-4
  let minDecay: Float = 1e-8
  if lossHistory.count < 2 { return }
  let window = Array(lossHistory.suffix(2))
  guard let previous = window.first else { return }
  guard let loss = window.last else { return }

  if loss <= previous * (1 - threshold) { return }
  let newLR = optimizer.learningRate * factor
  if optimizer.learningRate - newLR > minDecay {
    optimizer.learningRate = newLR
  }
}

WordSegCommand.main()
