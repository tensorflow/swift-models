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
import ModelSupport
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
  public var encoderEmbedding: Embedding<Float>
  public var encoderLSTM: LSTM<Float>

  // MARK: - Interpolation weight
  public var mlpInterpolation: MLP

  // MARK: - Lexical memory
  public var mlpMemory: MLP

  // MARK: - Character-level decoder
  public var decoderEmbedding: Embedding<Float>
  public var decoderLSTM: LSTM<Float>
  public var decoderDense: Dense<Float>

  // MARK: - Other layers
  public var dropout: Dropout<Float>

  // MARK: - Initializer
  public init(parameters: Parameters) {
    self.parameters = parameters

    // Encoder
    self.encoderEmbedding = Embedding(
      vocabularySize: parameters.chrVocab.count,
      embeddingSize: parameters.ndim)
    self.encoderLSTM = LSTM(
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
    self.decoderEmbedding = Embedding(
      vocabularySize: parameters.chrVocab.count,
      embeddingSize: parameters.ndim)
    self.decoderLSTM = LSTM(
      LSTMCell(
        inputSize: parameters.ndim,
        hiddenSize:
          parameters.ndim))
    self.decoderDense = Dense(inputSize: parameters.ndim, outputSize: parameters.chrVocab.count)

    // Other layers
    self.dropout = Dropout(probability: parameters.dropoutProb)
  }

  // MARK: - Encode
  /// Returns the hidden states of the encoder LSTM applied to the given sentence.
  public func encode(_ x: CharacterSequence) -> [Tensor<Float>] {
    var embedded = encoderEmbedding(x.tensor)
    embedded = dropout(embedded)
    let encoderStates = encoderLSTM(embedded.unstacked().differentiableMap { $0.rankLifted() })
    var encoderResult = Tensor(
      stacking: encoderStates.differentiableMap { $0.hidden.squeezingShape(at: 0) })
    encoderResult = dropout(encoderResult)
    return encoderResult.unstacked()
  }

  // MARK: - Decode
  /// Returns log probabilities for each of the candidates.
  public func decode(_ candidates: [CharacterSequence], _ state: Tensor<Float>) -> Tensor<Float> {
    // TODO(TF-433): Remove closure workaround when autodiff supports non-active rethrowing
    // functions (`Array.map`).
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
    var embeddedX = decoderEmbedding(x)
    embeddedX = dropout(embeddedX)

    // [batch x ndim]
    let stateBatch = state.rankLifted().tiled(multiples: Tensor([Int32(candidates.count), 1]))

    // [time] array of LSTM states whose `hidden` and `cell` fields have shape [batch x ndim]
    let decoderStates = decoderLSTM(
      embeddedX.unstacked(),
      initialState: LSTMCell.State(
        cell: Tensor(zeros: stateBatch.shape),
        hidden: stateBatch))

    // [time x batch x ndim]
    var decoderResult = Tensor(
      stacking: decoderStates.differentiableMap { $0.hidden })
    decoderResult = dropout(decoderResult)

    // [time x batch x chrVocab.count]
    let logits = decoderDense(decoderResult)

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
  func get_logp_lex(_ logp_lex: Tensor<Float>, _ candidate: CharacterSequence) -> Tensor<Float> {
    guard let index = parameters.strVocab.dictionary[candidate] else {
      return Tensor(-Float.infinity)
    }
    return logp_lex[Int(index)]
  }

  @differentiable
  public func buildLattice(_ sentence: CharacterSequence, maxLen: Int) -> Lattice {
    var lattice = Lattice(count: sentence.count)
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
        // TODO(TF-433): Use `Bool.&&` instead of nested if statements when autodiff supports
        // non-active rethrowing functions (`Bool.&&`).
        if candidate.count != 1 {
          if candidate.last == parameters.chrVocab.eos {
            // Prohibit strings such as ["t", "h", "e", "</s>"]
            continue
          }
        }
        candidates.append(candidate)
      }

      let current_state = states[pos]
      let logg = logg_batch[pos].identityADHack  // [2]
      let logp_lex = logp_lex_batch[pos].identityADHack  // [strVocab.chr.count]
      let logp_chr = decode(candidates, current_state).identityADHack  // [candidates.count]
      if pos != 0 {
        // Cleanup: lattice[pos].recomputeSemiringScore()
        var updatedNode = lattice[pos]
        updatedNode.recomputeSemiringScore()
        lattice.positions.update(at: pos, to: updatedNode)
      }

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

        // Cleanup: lattice[next_pos].edges.append(edge)
        var updatedNode = lattice[next_pos]
        updatedNode.edges.append(edge)
        lattice.positions.update(at: next_pos, to: updatedNode)
      }
    }

    // Cleanup: lattice[sentence.count].recomputeSemiringScore()
    var lastNode = lattice[sentence.count]
    lastNode.recomputeSemiringScore()
    lattice.positions.update(at: sentence.count, to: lastNode)

    return lattice
  }
}

extension Array {
  // NOTE(TF-1277): this mutating method exists as a workaround for `Array.subscript._modify` not
  // being differentiable.
  //
  // Semantically, it behaves like `Array.subscript.set`.
  @inlinable
  mutating func update(at index: Int, to value: Element) {
    self[index] = value
  }

  @usableFromInline
  @derivative(of: update)
  mutating func vjpUpdate(at index: Int, to value: Element) -> (
    value: (),
    pullback: (inout TangentVector) -> Element.TangentVector
  ) where Element: Differentiable {
    update(at: index, to: value)
    func pullback(_ dSelf: inout TangentVector) -> Element.TangentVector {
      let dElement = dSelf[index]
      dSelf.base[index] = dElement.zeroTangentVector
      return dElement
    }
    return ((), pullback)
  }
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

extension Tensor {
  // NOTE(TF-1008): this is a workaround for TF-1008 that is needed for differentiation
  // correctness.
  //
  // Remove this when differentiation uses per-instance zeros
  // (`Differentiable.zeroTangentVectorInitializer`) instead of static zeros
  // (`AdditiveArithmetic.zero`).
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  var identityADHack: Tensor {
    self
  }

  @derivative(of: identityADHack)
  func vjpIdentityADHack() -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) where Scalar: TensorFlowFloatingPoint {
    // In the pullback: capture only `self.shape`, not all of `self`.
    let shape = self.shape
    func pullback(_ v: Tensor) -> Tensor {
      if v.scalarCount == 1 {
        return v.broadcasted(to: shape)
      }
      return v
    }
    return (self, pullback)
  }
}