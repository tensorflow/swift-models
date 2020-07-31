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
    public var hiddenSize: Int

    /// The dropout rate.
    public var dropoutProbability: Double

    /// The union of characters used in this model.
    public var alphabet: Alphabet

    /// Contiguous sequences of characters encountered in the training data.
    public var lexicon: Lexicon

    /// The power of the length penalty.
    public var order: Int

    /// Creates an instance with `hiddenSize` units, `dropoutProbability`
    /// rate, `alphabet`, `lexicon`, and `order` power of length penalty.
    public init(
      hiddenSize: Int,
      dropoutProbability: Double,
      alphabet: Alphabet,
      lexicon: Lexicon,
      order: Int
    ) {
      self.hiddenSize = hiddenSize
      self.dropoutProbability = dropoutProbability
      self.alphabet = alphabet
      self.lexicon = lexicon
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
  public init(parameters: Parameters) {
    self.parameters = parameters

    // Encoder
    self.encoderEmbedding = Embedding(
      vocabularySize: parameters.alphabet.count,
      embeddingSize: parameters.hiddenSize)
    self.encoderLSTM = LSTM(
      LSTMCell(
        inputSize: parameters.hiddenSize,
        hiddenSize:
          parameters.hiddenSize))

    // Interpolation weight
    self.mlpInterpolation = MLP(
      inputSize: parameters.hiddenSize,
      hiddenSize: parameters.hiddenSize,
      outputSize: 2,
      dropoutProbability: parameters.dropoutProbability)

    // Lexical memory
    self.mlpMemory = MLP(
      inputSize: parameters.hiddenSize,
      hiddenSize: parameters.hiddenSize,
      outputSize: parameters.lexicon.count,
      dropoutProbability: parameters.dropoutProbability)

    // Character-level decoder
    self.decoderEmbedding = Embedding(
      vocabularySize: parameters.alphabet.count,
      embeddingSize: parameters.hiddenSize)
    self.decoderLSTM = LSTM(
      LSTMCell(
        inputSize: parameters.hiddenSize,
        hiddenSize:
          parameters.hiddenSize))
    self.decoderDense = Dense(
      inputSize: parameters.hiddenSize, outputSize: parameters.alphabet.count)

    // Other layers
    self.dropout = Dropout(probability: parameters.dropoutProbability)
  }

  // MARK: - Encode

  /// Returns the hidden states of the encoder LSTM applied to `x`, using
  /// `device`.
  public func encode(_ x: CharacterSequence, device: Device) -> [Tensor<Float>] {
    let embedded = dropout(encoderEmbedding(x.tensor(device: device)))
    let encoderStates = encoderLSTM(embedded.unstacked().differentiableMap { $0.rankLifted() })
    let encoderResult = dropout(Tensor(
      stacking: encoderStates.differentiableMap { $0.hidden.squeezingShape(at: 0) }))
    return encoderResult.unstacked()
  }

  // MARK: - Decode

  /// Returns the log probabilities for each sequence in `candidates`, given
  /// hidden `state` from the encoder LSTM, using `device`.
  public func decode(_ candidates: [CharacterSequence], _ state: Tensor<Float>, device: Device)
    -> Tensor<Float>
  {
    // TODO(TF-433): Remove closure workaround when autodiff supports non-active rethrowing
    // functions (`Array.map`).
    let maxLen = { candidates.map { $0.count }.max()! + 1 }()
    var xBatch: [Int32] = []
    var yBatch: [Int32] = []
    for candidate in candidates {
      let padding = Array(repeating: parameters.alphabet.pad, count: maxLen - candidate.count - 1)

      // x is </w>{sentence}{padding}
      xBatch.append(parameters.alphabet.eow)
      xBatch.append(contentsOf: candidate.characters)
      xBatch.append(contentsOf: padding)

      // y is {sentence}</w>{padding}
      yBatch.append(contentsOf: candidate.characters)
      yBatch.append(parameters.alphabet.eow)
      yBatch.append(contentsOf: padding)
    }

    // Shapes are [time x batch] so that we can unstack the time dimension into the array that
    // the LSTM wants as input.
    let x: Tensor<Int32> = Tensor(
      shape: [candidates.count, maxLen], scalars: xBatch, on: device
    ).transposed()
    let y: Tensor<Int32> = Tensor(
      shape: [candidates.count, maxLen], scalars: yBatch, on: device
    ).transposed()

    // [time x batch x hiddenSize]
    let embeddedX = dropout(decoderEmbedding(x))

    // [batch x hiddenSize]
    let stateBatch = state.rankLifted().tiled(multiples: [candidates.count, 1])

    // [time] array of LSTM states whose `hidden` and `cell` fields have shape [batch x hiddenSize]
    let decoderStates = decoderLSTM(
      embeddedX.unstacked(),
      initialState: LSTMCell.State(
        cell: Tensor(zeros: stateBatch.shape, on: device),
        hidden: stateBatch))

    // [time x batch x hiddenSize]
    let decoderResult = dropout(Tensor(
      stacking: decoderStates.differentiableMap { $0.hidden }))

    // [time x batch x alphabet.count]
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
    let padScalars = [Int32](repeating: parameters.alphabet.pad, count: candidates.count * maxLen)
    let noPad = Tensor<Int32>(
      y .!= Tensor(shape: y.shape, scalars: padScalars, on: device))
    let noPadFloat = Tensor<Float>(noPad)
    let logpExcludingPad = logp * noPadFloat

    // [batch]
    let candidateLogP = logpExcludingPad.transposed().sum(squeezingAxes: 1)

    return candidateLogP
  }

  // MARK: - buildLattice

  /// Returns the log probability for `candidate` from the lexical memory
  /// `logp_lex`.
  func get_logp_lex(_ logp_lex: [Float], _ candidate: CharacterSequence) -> Float {
    guard let index = parameters.lexicon.dictionary[candidate] else {
      return -Float.infinity
    }
    return logp_lex[Int(index)]
  }

  /// Returns a lattice for `sentence` with `maxLen` maximum sequence length.
  @differentiable
  public func buildLattice(_ sentence: CharacterSequence, maxLen: Int, device: Device) -> Lattice {
    var lattice = Lattice(count: sentence.count)
    let states = encode(sentence, device: device)
    let logg_batch = mlpInterpolation(Tensor(stacking: states))
    let logp_lex_batch = mlpMemory(Tensor(stacking: states))
    for pos in 0..<sentence.count {
      var candidates: [CharacterSequence] = []
      for span in 1..<min(sentence.count - pos + 1, maxLen + 1) {
        // TODO: avoid copies?
        let candidate =
          CharacterSequence(
            alphabet: parameters.alphabet,
            characters: sentence[pos..<pos + span])
        // TODO(TF-433): Use `Bool.&&` instead of nested if statements when autodiff supports
        // non-active rethrowing functions (`Bool.&&`).
        if candidate.count != 1 {
          if candidate.last == parameters.alphabet.eos {
            // Prohibit strings such as ["t", "h", "e", "</s>"]
            continue
          }
        }
        candidates.append(candidate)
      }

      let current_state = states[pos]
      let logg = logg_batch[pos].scalarsADHack(device: device)  // [2]
      let logp_lex = logp_lex_batch[pos].scalarsADHack(device: device)  // [strVocab.chr.count]
      let logp_chr = decode(candidates, current_state, device: device)
        .scalarsADHack(device: device)  // [candidates.count]
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

      LazyTensorBarrier()
    }

    // Cleanup: lattice[sentence.count].recomputeSemiringScore()
    var lastNode = lattice[sentence.count]
    lastNode.recomputeSemiringScore()
    lattice.positions.update(at: sentence.count, to: lastNode)

    return lattice
  }
}

extension Array {

  /// Sets the `index`th element of `self` to `value`.
  ///
  /// Semantically, this function behaves like `Array.subscript.set`.
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
    let elementZero = self[index].zeroTangentVector
    update(at: index, to: value)
    func pullback(_ dSelf: inout TangentVector) -> Element.TangentVector {
      let dElement = dSelf[index]
      dSelf.base[index] = elementZero
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

  /// Creates an instance with `inputSize`, `hiddenSize`,
  /// `dropoutProbability`, and `outputSize`.
  public init(inputSize: Int, hiddenSize: Int, outputSize: Int, dropoutProbability: Double) {
    dense1 = Dense(inputSize: inputSize, outputSize: hiddenSize, activation: tanh)
    dropout = Dropout(probability: dropoutProbability)
    dense2 = Dense(inputSize: hiddenSize, outputSize: outputSize, activation: logSoftmax)
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
  func scalarsADHack(device: Device) -> [Scalar] {
    scalars
  }

  /// Returns the value and pullback of `self.scalarsADHack`.
  @derivative(of: scalarsADHack)
  func vjpScalarsADHack(device: Device) -> (
    value: [Scalar], pullback: (Array<Scalar>.TangentVector) -> Tensor
  ) where Scalar: TensorFlowFloatingPoint {
    // In the pullback: capture only `self.shape`, not all of `self`.
    let shape = self.shape
    func pullback(_ tv: Array<Scalar>.TangentVector) -> Tensor {
      if tv.count == 0 {
        return Tensor(zeros: shape, on: device)
      }
      return Tensor(shape: shape, scalars: tv.base, on: device)
    }
    return (scalars, pullback)
  }
}
