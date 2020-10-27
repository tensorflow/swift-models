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

import Benchmark
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

let WordSegScore = wordSegSuite(
  name: "WordSegScore",
  operation: score)

let WordSegScoreAndGradient = wordSegSuite(
  name: "WordSegScoreAndGradient",
  operation: scoreAndGradient)

let WordSegViterbi = wordSegSuite(
  name: "WordSegViterbi",
  operation: viterbi)

let maximumSequenceLength = 18

func wordSegSuite(name: String, operation: @escaping (SNLM, CharacterSequence, Device) -> Void)
  -> BenchmarkSuite
{
  let function = wordSegBenchmark(operation)
  let columns = Columns([
    "name",
    "wall_time",
    "startup_time",
    "iterations",
    "time_median",
    "time_min",
    "time_max",
  ])
  return BenchmarkSuite(name: name, settings: WarmupIterations(10), columns) { suite in
    suite.benchmark(
      "sentence_4", settings: Length(4), Backend(.eager), function: function)
    suite.benchmark(
      "sentence_4_x10", settings: Length(4), Backend(.x10), function: function)
    suite.benchmark(
      "sentence_8", settings: Length(8), Backend(.eager), function: function)
    suite.benchmark(
      "sentence_8_x10", settings: Length(8), Backend(.x10), function: function)
    suite.benchmark(
      "sentence_14", settings: Length(14), Backend(.eager), function: function)
    suite.benchmark(
      "sentence_14_x10", settings: Length(14), Backend(.x10), function: function)
  }
}

func wordSegBenchmark(_ operation: @escaping (SNLM, CharacterSequence, Device) -> Void) -> (
  (inout BenchmarkState) throws -> Void
) {
  return { state in
    let settings = state.settings
    let device = settings.device
    let length = settings.length!

    state.start()

    let dataset: WordSegDataset
    if let trainingFilePath = settings.datasetFilePath {
      dataset = try WordSegDataset(training: trainingFilePath)
    } else {
      dataset = try WordSegDataset()
    }

    let sentence = try testSentence(
      length: length,
      alphabet: dataset.alphabet)

    // Model settings are drawn from known benchmarks.
    let lexicon = Lexicon(
      from: [sentence],
      alphabet: dataset.alphabet,
      maxLength: maximumSequenceLength,
      minFrequency: 10
    )

    let modelParameters = SNLM.Parameters(
      hiddenSize: 512,
      dropoutProbability: 0.5,
      alphabet: dataset.alphabet,
      lexicon: lexicon,
      order: 5
    )

    var model = SNLM(parameters: modelParameters)
    model.move(to: device)

    while true {
      operation(model, sentence, device)
      LazyTensorBarrier()

      do {
        try state.end()
      } catch {
        if settings.backend == .x10 {
          // A synchronous barrier is needed for X10 to ensure all execution completes
          // before tearing down the model.
          LazyTensorBarrier(wait: true)
        }
        throw error
      }
      state.start()
    }
  }
}

func testSentence(length: Int, alphabet: Alphabet) throws -> CharacterSequence {
  let sourceSentence = [
    "you", "like", "daddy's", "whiskers", "just", "gonna", "eat", "the",
    "comb", "and", "that's", "all",
  ]

  let truncatedSentence = sourceSentence.prefix(length).reduce("", +)  // + ["</s"]
  return try CharacterSequence(alphabet: alphabet, appendingEoSTo: truncatedSentence)
}

func score(model: SNLM, sentence: CharacterSequence, device: Device) {
  let lattice = model.buildLattice(sentence, maxLen: maximumSequenceLength, device: device)
  let score = lattice[sentence.count].semiringScore
  let _ = score.logr + score.logp
}

func scoreAndGradient(model: SNLM, sentence: CharacterSequence, device: Device) {
  let lambd: Float = 0.00075

  let _ = valueWithGradient(at: model) { model -> Tensor<Float> in
    let lattice = model.buildLattice(sentence, maxLen: maximumSequenceLength, device: device)
    let score = lattice[sentence.count].semiringScore
    let expectedLength = exp(score.logr - score.logp)
    let loss = -1 * score.logp + lambd * expectedLength
    return Tensor(loss, on: device)
  }
}

func viterbi(model: SNLM, sentence: CharacterSequence, device: Device) {
  var lattice = model.buildLattice(sentence, maxLen: maximumSequenceLength, device: device)
  let _ = lattice.viterbi(sentence: sentence)
}
