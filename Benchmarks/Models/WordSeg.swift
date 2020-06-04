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

// We're repurposing `batchSize` for the sequence length parameter in the WordSeg benchmarks.
// Typical sizes are 4, 8, and 14.
enum WordSegScore: BenchmarkModel {
    static var name: String { "WordSegScore" }

    static func defaults(for variety: BenchmarkVariety) -> BenchmarkSettings {
        return BenchmarkSettings(
            duration: .batches(20), batchSize: 4, iterations: 1,
            warmupBatches: 10, synthetic: false, backend: .eager)
    }

    static func makeInferenceBenchmark(settings: BenchmarkSettings) -> Benchmark {
        return WordSegBenchmark(settings: settings, operation: WordSegBenchmark.score)
    }

    static func makeTrainingBenchmark(settings: BenchmarkSettings) -> Benchmark {
        return WordSegBenchmark(settings: settings, operation: WordSegBenchmark.score)
    }
}

enum WordSegScoreAndGradient: BenchmarkModel {
    static var name: String { "WordSegScoreAndGradient" }

    static func defaults(for variety: BenchmarkVariety) -> BenchmarkSettings {
        return BenchmarkSettings(
            duration: .batches(20), batchSize: 4, iterations: 1,
            warmupBatches: 10, synthetic: false, backend: .eager)
    }

    static func makeInferenceBenchmark(settings: BenchmarkSettings) -> Benchmark {
        return WordSegBenchmark(settings: settings, operation: WordSegBenchmark.scoreAndGradient)
    }

    static func makeTrainingBenchmark(settings: BenchmarkSettings) -> Benchmark {
        return WordSegBenchmark(settings: settings, operation: WordSegBenchmark.scoreAndGradient)
    }
}

enum WordSegViterbi: BenchmarkModel {
    static var name: String { "WordSegViterbi" }

    static func defaults(for variety: BenchmarkVariety) -> BenchmarkSettings {
        return BenchmarkSettings(
            duration: .batches(20), batchSize: 4, iterations: 1,
            warmupBatches: 10, synthetic: false, backend: .eager)
    }

    static func makeInferenceBenchmark(settings: BenchmarkSettings) -> Benchmark {
        return WordSegBenchmark(settings: settings, operation: WordSegBenchmark.viterbi)
    }

    static func makeTrainingBenchmark(settings: BenchmarkSettings) -> Benchmark {
        return WordSegBenchmark(settings: settings, operation: WordSegBenchmark.viterbi)
    }
}

let maximumSequenceLength = 18

struct WordSegBenchmark: Benchmark {
    let batchSize: Int
    let duration: BenchmarkDuration
    let operation: (SNLM, CharacterSequence) -> ()

    init(settings: BenchmarkSettings, operation: @escaping (SNLM, CharacterSequence) -> ()) {
        self.duration = settings.duration
        self.batchSize = settings.batchSize
        self.operation = operation
    }

    func run(backend: Backend) -> [Double] {
        let device: Device
        switch backend {
        case .eager: device = Device.defaultTFEager
        case .x10: device = Device.defaultXLA
        }
        
        var batchTimings: [Double] = []
        var beforeBatch = timestampInMilliseconds()

        do {
            let dataset = try WordSegDataset()
            let sentence = try WordSegBenchmark.testSentence(length: batchSize,
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
            
            let iterations: Int
            switch duration {
            case let .batches(value):
                iterations = value
            case let .epochs(value):
                iterations = value
            }
            
            for _ in 0..<iterations {
                operation(model, sentence)
                LazyTensorBarrier()
                
                batchTimings.append(durationInMilliseconds(since: beforeBatch))
                beforeBatch = timestampInMilliseconds()
            }
        } catch {
            fatalError("Error during WordSeg benchmark: \(error)")
        }
        
        return batchTimings
    }
}

extension WordSegBenchmark {
    static func testSentence(length: Int, alphabet: Alphabet) throws -> CharacterSequence {
        let sourceSentence = ["you", "like", "daddy's", "whiskers", "just", "gonna", "eat", "the",
            "comb", "and", "that's", "all"]
        
        let truncatedSentence = sourceSentence.prefix(length).reduce("", +) // + ["</s"]
        return try CharacterSequence(alphabet: alphabet, appendingEoSTo: truncatedSentence)
    }

    static func score(model: SNLM, sentence: CharacterSequence) {
        let lattice = model.buildLattice(sentence, maxLen: maximumSequenceLength)
        let score = lattice[sentence.count].semiringScore
        let _ = score.logr + score.logp
    }

    static func scoreAndGradient(model: SNLM, sentence: CharacterSequence) {
        let lambd: Float = 0.00075

        let _ = valueWithGradient(at: model) { model -> Float in
          let lattice = model.buildLattice(sentence, maxLen: maximumSequenceLength)
          let score = lattice[sentence.count].semiringScore
          let expectedLength = exp(score.logr - score.logp)
          let loss = -1 * score.logp + lambd * expectedLength
          return loss
        }
    }

    static func viterbi(model: SNLM, sentence: CharacterSequence) {
        var lattice = model.buildLattice(sentence, maxLen: maximumSequenceLength)
        let _ = lattice.viterbi(sentence: sentence)
    }
}
