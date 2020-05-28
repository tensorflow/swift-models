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

/// A Segmental Neural Language Model for word segmentation, as described in
/// the above paper.
public struct SNLM: EuclideanDifferentiable, KeyPathIterable {
  /// A set of configuration parameters that define model behavior.
  public struct Parameters {
    /// The hidden unit size.
    public var ndim: Int
    /// The dropout rate.
    public var dropoutProb: Double
    /// The character vocabulary.
    public var chrVocab: Alphabet
    /// The string vocabulary.
    public var strVocab: Lexicon
    /// The power of the length penalty.
    public var order: Int

    /// Creates an instance with `ndim` hidden units, `dropoutProb` dropout
    /// rate, `chrVocab` alphabet, `strVocab` lexicon, and `order` power of
    /// length penalty.
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

  /// The configuration parameters that define model behavior.
  @noDerivative public var parameters: Parameters

  // MARK: - Encoder
  /// The embedding layer for the encoder.
  public var encoderEmbedding: Embedding<Float>
  /// The LSTM layer for the encoder.
  public var encoderLSTM: LSTM<Float>

  // MARK: - Interpolation weight
  /// The interpolation weight, which determines the proportion of
  /// contributions from the lexical memory and character generation.
  public var mlpInterpolation: MLP

  // MARK: - Lexical memory
  /// The lexical memory.
  public var mlpMemory: MLP

  // MARK: - Character-level decoder
  /// The embedding layer for the decoder.
  public var decoderEmbedding: Embedding<Float>
  /// The LSTM layer for the decoder.
  public var decoderLSTM: LSTM<Float>
  /// The dense layer for the decoder.
  public var decoderDense: Dense<Float>

  // MARK: - Other layers
  /// The dropout layer for both the encoder and decoder.
  public var dropout: Dropout<Float>

  // MARK: - Initializer
  /// Creates an instance with the configuration defined by `parameters`.
  ///
  /// - Parameter parameters: the model configuration.
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
  /// Returns the hidden states of the encoder LSTM applied to `x`.
  ///
  /// - Parameter x: the character sequence to encode.
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
  /// Returns the log probabilities for each of the candidates.
  ///
  /// - Parameter candidates: the character sequences to decode.
  /// - Parameter state: the hidden state from the encoder LSTM.
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
  /// Returns the log likelihood for `candidate` from the lexical memory
  /// `logp_lex`.
  ///
  /// - Parameter logp_lex: all log likelihoods in the lexical memory.
  /// - Parameter candidate: the character sequence for which to retrieve the
  ///   log likelihood.
  func get_logp_lex(_ logp_lex: Tensor<Float>, _ candidate: CharacterSequence) -> Tensor<Float> {
    guard let index = parameters.strVocab.dictionary[candidate] else {
      return Tensor(-Float.infinity)
    }
    return logp_lex[Int(index)]
  }

  /// Returns a complete lattice for `sentence` with a maximum length of
  /// `maxLen`.
  ///
  /// - Parameter sentence: the character sequence used for determining
  ///   segmentation.
  /// - Parameter maxLen: the maximum allowable sequence length.
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
  /// Sets the `index`th element of `self` to `value`. Semantically, it
  /// behaves like `Array.subscript.set`.
  ///
  /// - Note: this mutating method exists as a workaround for
  ///   `Array.subscript._modify` not being differentiable (TF-1277).
  @inlinable
  mutating func update(at index: Int, to value: Element) {
    self[index] = value
  }

  /// Returns the value and pullback of `self.update`.
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

/// A multilayer perceptron with three layers.
public struct MLP: Layer {
  /// The first dense layer.
  public var dense1: Dense<Float>
  /// The dropout layer.
  public var dropout: Dropout<Float>
  /// The second dense layer.
  public var dense2: Dense<Float>

  /// Creates an instance with input size `nIn`, `nHidden` hidden units,
  /// dropout probability `dropoutProbability` and output size `nOut`.
  ///
  /// - Parameter nIn: input size.
  /// - Parameter nHidden: number of hidden units.
  /// - Parameter nOut: output size.
  /// - Parameter dropoutProbability: probability that an input is dropped.
  public init(nIn: Int, nHidden: Int, nOut: Int, dropoutProbability: Double) {
    dense1 = Dense(inputSize: nIn, outputSize: nHidden, activation: tanh)
    dropout = Dropout(probability: dropoutProbability)
    dense2 = Dense(inputSize: nHidden, outputSize: nOut, activation: logSoftmax)
  }

  /// Returns the result of applying all three layers in sequence to `input`.
  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    return dense2(dropout(dense1(input)))
  }
}

extension Tensor {
  /// Returns `self`.
  ///
  /// - Note: this is a workaround for TF-1008 that is needed for
  /// differentiation correctness.
  // TODO: Remove this when differentiation uses per-instance zeros
  // (`Differentiable.zeroTangentVectorInitializer`) instead of static zeros
  // (`AdditiveArithmetic.zero`).
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  var identityADHack: Tensor {
    self
  }

  /// Returns the value and pullback of `self.identityADHack`.
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
