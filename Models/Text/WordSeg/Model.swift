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

// Original Paper:
// "Learning to Discover, Ground, and Use Words with Segmental Neural Language
// Models"
// Kazuya Kawakami, Chris Dyer, Phil Blunsom
// https://www.aclweb.org/anthology/P19-1645.pdf
// This implementation is not affiliated with DeepMind and has not been
// verified by the authors.

import TensorFlow

/// SNLM
///
/// A representation of the Segmental Neural Language Model.
///
/// \ref https://www.aclweb.org/anthology/P19-1645.pdf
public struct SNLM: EuclideanDifferentiable, KeyPathIterable {
  public struct Parameters {
    public var ndim: Int
    public var dropoutProb: Double
    public var chrVocab: Alphabet
    public var strVocab: Lexicon
    public var order: Int

    public init(
      ndim: Int,
      dropoutProb: Double,
      chrVocab: Alphabet,
      strVocab: Lexicon,
      order: Int
    ) {
      self.ndim = ndim
      self.dropoutProb = dropoutProb
      self.chrVocab = chrVocab
      self.strVocab = strVocab
      self.order = order
    }
  }

  @noDerivative public var parameters: Parameters

  // MARK: - Encoder

  public var embEnc: Embedding<Float>
  public var lstmEnc: LSTM<Float>

  // MARK: - Interpolation weight

  public var mlpInterpolation: MLP

  // MARK: - Lexical memory

  public var mlpMemory: MLP

  // MARK: - Character-level decoder

  public var embDec: Embedding<Float>
  public var lstmDec: LSTM<Float>
  public var denseDec: Dense<Float>

  // MARK: - Other layers

  public var drop: Dropout<Float>

  // MARK: - Initializer

  public init(parameters: Parameters) {
    self.parameters = parameters

    // Encoder
    self.embEnc = Embedding(
      vocabularySize: parameters.chrVocab.count,
      embeddingSize: parameters.ndim)
    self.lstmEnc = LSTM(
      LSTMCell(
        inputSize: parameters.ndim,
        hiddenSize:
          parameters.ndim))

    // Interpolation weight
    self.mlpInterpolation = MLP(
      nIn: parameters.ndim,
      nHidden: parameters.ndim,
      nOut: 2,
      dropoutProbability: parameters.dropoutProb)

    // Lexical memory
    self.mlpMemory = MLP(
      nIn: parameters.ndim,
      nHidden: parameters.ndim,
      nOut: parameters.strVocab.count,
      dropoutProbability: parameters.dropoutProb)

    // Character-level decoder
    self.embDec = Embedding(
      vocabularySize: parameters.chrVocab.count,
      embeddingSize: parameters.ndim)
    self.lstmDec = LSTM(
      LSTMCell(
        inputSize: parameters.ndim,
        hiddenSize:
          parameters.ndim))
    self.denseDec = Dense(inputSize: parameters.ndim, outputSize: parameters.chrVocab.count)

    // Other layers
    self.drop = Dropout(probability: parameters.dropoutProb)
  }

  // MARK: - Encode

  /// Returns the hidden states of the encoder LSTM applied to the given sentence.
  public func encode(_ x: CharacterSequence) -> [Tensor<Float>] {
    let embedded = drop(embEnc(x.tensor))
    let encoderStates = lstmEnc(embedded.unstacked().differentiableMap { $0.rankLifted() })
    return encoderStates.differentiableMap { $0.hidden.squeezingShape(at: 0) }
  }

  // MARK: - Decode

  /// Returns log probabilities for each of the candidates.
  public func decode(_ candidates: [CharacterSequence], _ state: Tensor<Float>) -> Tensor<Float> {
    // TODO: Shouldn't use a closure here.
    let maxLen = { candidates.map { $0.count }.max()! + 1 }()
    var xBatch: [Int32] = []
    var yBatch: [Int32] = []
    for candidate in candidates {
      let padding = Array(repeating: parameters.chrVocab.pad, count: maxLen - candidate.count - 1)

      // x is </w>{sentence}{padding}
      xBatch.append(parameters.chrVocab.eow)
      xBatch.append(contentsOf: candidate.characters)
      xBatch.append(contentsOf: padding)

      // y is {sentence}</w>{padding}
      yBatch.append(contentsOf: candidate.characters)
      yBatch.append(parameters.chrVocab.eow)
      yBatch.append(contentsOf: padding)
    }

    // Shapes are [time x batch] so that we can unstack the time dimension into the array that
    // the LSTM wants as input.
    let x: Tensor<Int32> = Tensor(shape: [candidates.count, maxLen], scalars: xBatch).transposed()
    let y: Tensor<Int32> = Tensor(shape: [candidates.count, maxLen], scalars: yBatch).transposed()

    // [time x batch x ndim]
    let embeddedX = drop(embDec(x))

    // [batch x ndim]
    let stateBatch = state.rankLifted().tiled(multiples: Tensor([Int32(candidates.count), 1]))

    // [time] array of LSTM states whose `hidden` and `cell` fields have shape [batch x ndim]
    let decoderStates = lstmDec(
      embeddedX.unstacked(),
      initialState: LSTMCell.State(
        cell: Tensor(zeros: stateBatch.shape),
        hidden: stateBatch))

    // [time x batch x ndim]
    // TODO: Need to add dropout here, but it breaks AD.
    let decoderResult = Tensor(
      stacking: decoderStates.differentiableMap { $0.hidden })

    // [time x batch x chrVocab.count]
    let logits = denseDec(decoderResult)

    // [time x batch]
    let logp =
      -1
      * softmaxCrossEntropy(
        logits: logits.reshaped(to: [logits.shape[0] * logits.shape[1], logits.shape[2]]),
        labels: y.flattened(),
        reduction: identity
      ).reshaped(to: y.shape)

    // [time x batch]
    let logpExcludingPad = logp * Tensor<Float>(y .!= parameters.chrVocab.pad)

    // [batch]
    let candidateLogP = logpExcludingPad.transposed().sum(squeezingAxes: 1)

    return candidateLogP
  }

  // MARK: - buildLattice

  //def get_logp_lex(self, logp_lex, candidate):
  //  if candidate in self.str_vocab:
  //    candidate_idx = self.str_vocab[candidate]
  //    return logp_lex[candidate_idx]
  //  else:
  //    return torch.log(torch_util.var_from_scaler(0.0, "FloatTensor", self.gpu))

  func get_logp_lex(_ logp_lex: [Float], _ candidate: CharacterSequence) -> Float {
    guard let index = parameters.strVocab.dictionary[candidate] else {
      return -Float.infinity
    }
    return logp_lex[Int(index)]
  }

  // TODO: Triggers compiler crash.
  @differentiable
  public func buildLattice(_ sentence: CharacterSequence, maxLen: Int) -> Lattice {
    var lattice = Lattice(count: sentence.count, embEnc.embeddings)
    let states = encode(sentence)
    let logg_batch = mlpInterpolation(Tensor(stacking: states))
    let logp_lex_batch = mlpMemory(Tensor(stacking: states))
    for pos in 0..<sentence.count {
      var candidates: [CharacterSequence] = []
      for span in 1..<min(sentence.count - pos + 1, maxLen + 1) {
        // TODO: avoid copies?
        let candidate =
          CharacterSequence(
            alphabet: parameters.chrVocab,
            characters: sentence[pos..<pos + span])
        // TODO: use && instead of nested ifs (AD workaround)
        if candidate.count != 1 {
          if candidate.last == parameters.chrVocab.eos {
            // Prohibit strings such as ["t", "h", "e", "</s>"]
            continue
          }
        }
        candidates.append(candidate)
      }

      //# Calculate probabilities
      //current_state = states[pos]
      //logg = logg_batch[pos]
      //logp_lex = logp_lex_batch[pos]
      //logp_chr = self.decode(candidates, current_state)
      //# Update semiring score
      //if pos != 0:
      //  lattice[pos]["semiring_score"] = semiring.add(lattice[pos]["edges"],
      //                                                self.gpu)
      let current_state = states[pos]
      let logg = scalarsWithADHack(logg_batch[pos])  // [2]
      let logp_lex = scalarsWithADHack(logp_lex_batch[pos])  // [strVocab.chr.count]
      let logp_chr = scalarsWithADHack(decode(candidates, current_state))  // [candidates.count]
      if pos != 0 {
        // TODO: Mutate in place when AD supports it.
        let updatedNode = Lattice.Node(
          bestEdge: lattice[pos].bestEdge,
          bestScore: lattice[pos].bestScore,
          edges: lattice[pos].edges,
          semiringScore: lattice[pos].computeSemiringScore()
        )
        lattice.positions = update(lattice.positions, index: pos, value: updatedNode)
      }

      //for i, candidate in enumerate(candidates):
      //  next_pos = pos + len(candidate)
      //  logp_lex_i = self.get_logp_lex(logp_lex, candidate)
      //  logp_chr_i = logp_chr[i]
      //  logp_i = torch_util.logsumexp([logg[0] + logp_lex_i,
      //                                 logg[1] + logp_chr_i])
      //  # Create an edge.
      //  edge = make_edge(pos, next_pos, candidate, logp_i,
      //                   lattice[pos]["semiring_score"], self.order, self.gpu)
      //  lattice[next_pos]["edges"].append(edge)

      for (i, candidate) in candidates.enumerated() {
        let next_pos = pos + candidate.count
        let logp_lex_i = get_logp_lex(logp_lex, candidate)
        let logp_chr_i = logp_chr[i]
        let logp_i = logSumExp(logg[0] + logp_lex_i, logg[1] + logp_chr_i)
        let edge = Lattice.Edge(
          start: pos,
          end: next_pos,
          sentence: candidate,
          logp: logp_i,
          previous: lattice[pos].semiringScore,
          order: parameters.order)

        // TODO: Mutate in place when AD supports it.
        let updatedNode = Lattice.Node(
          bestEdge: lattice[next_pos].bestEdge,
          bestScore: lattice[next_pos].bestScore,
          edges: lattice[next_pos].edges + [edge],
          semiringScore: lattice[next_pos].semiringScore
        )
        lattice.positions = update(lattice.positions, index: next_pos, value: updatedNode)
      }
    }

    //lattice[sentence.count].recomputeSemiringScore()
    // TODO: Mutate in place when AD supports it.
    let updatedNode = Lattice.Node(
      bestEdge: lattice[sentence.count].bestEdge,
      bestScore: lattice[sentence.count].bestScore,
      edges: lattice[sentence.count].edges,
      semiringScore: lattice[sentence.count].computeSemiringScore()
    )
    lattice.positions = update(lattice.positions, index: sentence.count, value: updatedNode)

    return lattice
  }
}

func update<T>(_ arr: [T], index: Int, value: T) -> [T] {
  var m = arr
  m[index] = value
  return m
}

@derivative(of: update)
func vjpupdate<T: Differentiable>(_ arr: [T], index: Int, value: T) -> (
  value: [T],
  pullback: (Array<T>.TangentVector) -> (Array<T>.TangentVector, T.TangentVector)
) {
  func pullback(_ tv: Array<T>.TangentVector) -> (Array<T>.TangentVector, T.TangentVector) {
    var m = tv
    m[index] = T.TangentVector.zero
    return (m, tv[index])
  }
  return (update(arr, index: index, value: value), pullback)
}

public struct MLP: Layer {
  public var dense1: Dense<Float>
  public var dropout: Dropout<Float>
  public var dense2: Dense<Float>

  public init(nIn: Int, nHidden: Int, nOut: Int, dropoutProbability: Double) {
    dense1 = Dense(inputSize: nIn, outputSize: nHidden, activation: tanh)
    dropout = Dropout(probability: dropoutProbability)
    dense2 = Dense(inputSize: nHidden, outputSize: nOut, activation: logSoftmax)
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    return dense2(dropout(dense1(input)))
  }
}

// TODO: Better way of dealing with this problem.
func scalarsWithADHack(_ t: Tensor<Float>) -> [Float] {
  t.scalars
}

@derivative(of: scalarsWithADHack)
func vjpScalarsHack(_ t: Tensor<Float>) -> (
  value: [Float], pullback: (Array<Float>.TangentVector) -> Tensor<Float>
) {
  // TODO: Capture less stuff.
  func pullback(_ tv: Array<Float>.TangentVector) -> Tensor<Float> {
    if tv.count == 0 {
      return Tensor(zeros: t.shape)
    }
    return Tensor(shape: t.shape, scalars: tv.base)
  }
  return (t.scalars, pullback)
}
