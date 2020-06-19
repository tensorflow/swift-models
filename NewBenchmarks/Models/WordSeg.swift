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

let WordSeg = BenchmarkSuite(name: "WordSeg") { suite in
  // Typical sizes for sequence length parameter in the WordSeg benchmarks: 4, 8, and 14.
  let settings: [BenchmarkSetting] = [Length(4), WarmupIterations(10)]

  suite.benchmark("score", settings: settings) { state in
    let device = state.settings.device
    try runWordSegBenchmark(state: &state) { model, sentence in
      let lattice = model.buildLattice(sentence, maxLen: maximumSequenceLength, device: device)
      let score = lattice[sentence.count].semiringScore
      let _ = score.logr + score.logp
    }
  }

  suite.benchmark("score_and_gradient", settings: settings) { state in
    let device = state.settings.device
    try runWordSegBenchmark(state: &state) { model, sentence in
      let lambd: Float = 0.00075

      let _ = valueWithGradient(at: model) { model -> Tensor<Float> in
        let lattice = model.buildLattice(sentence, maxLen: maximumSequenceLength, device: device)
        let score = lattice[sentence.count].semiringScore
        let expectedLength = exp(score.logr - score.logp)
        let loss = -1 * score.logp + lambd * expectedLength
        return Tensor(loss, on: device)
      }
    }
  }

  suite.benchmark("viterbi", settings: settings) { state in
    let device = state.settings.device
    try runWordSegBenchmark(state: &state) { model, sentence in
      var lattice = model.buildLattice(sentence, maxLen: maximumSequenceLength, device: device)
      let _ = lattice.viterbi(sentence: sentence)
    }
  }
}

let maximumSequenceLength = 18

func runWordSegBenchmark(
  state: inout BenchmarkState, operation: (SNLM, CharacterSequence) -> Void
) throws {
  let settings = state.settings
  let device = settings.device
  let length = settings.length!

  state.start()

  let dataset = try WordSegDataset()
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
    operation(model, sentence)
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

func testSentence(length: Int, alphabet: Alphabet) throws -> CharacterSequence {
  let sourceSentence = [
    "you", "like", "daddy's", "whiskers", "just", "gonna", "eat", "the",
    "comb", "and", "that's", "all",
  ]

  let truncatedSentence = sourceSentence.prefix(length).reduce("", +)  // + ["</s"]
  return try CharacterSequence(alphabet: alphabet, appendingEoSTo: truncatedSentence)
}
