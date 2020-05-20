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

import ModelSupport
import TensorFlow
import TextModels
import XCTest

extension SNLM {
  /// Sets the model parameters to the given parameters exported from the model.
  mutating func setParameters(_ p: SNLMParameters) {
    setEmbedding(&encoderEmbedding, to: p.emb_enc)
    setLSTM(&encoderLSTM.cell, to: p.lstm_enc)
    setMLP(&mlpInterpolation, to: p.mlp_interpolation)
    setMLP(&mlpMemory, to: p.mlp_memory)
    setEmbedding(&decoderEmbedding, to: p.emb_dec)
    setLSTM(&decoderLSTM.cell, to: p.lstm_dec)
    setDense(&decoderDense, to: p.linear_dec)
  }

  private func checkShapeAndSet(_ tensor: inout Tensor<Float>, to value: Tensor<Float>) {
    assert(
      tensor.shape == value.shape, "shape mismatch while setting: \(tensor.shape) to \(value.shape)"
    )
    tensor = value
  }

  // Sets the given Embedding parameters to the given embedding parameters.
  private func setEmbedding(_ embedding: inout Embedding<Float>, to p: EmbeddingParameters) {
    checkShapeAndSet(&embedding.embeddings, to: p.weight)
  }

  /// Sets the given LSTM cell's parameters to the given LSTM parameters.
  private func setLSTM(_ lstm: inout LSTMCell<Float>, to p: LSTMParameters) {
    let fusedWeight = p.weight_ih_l0.concatenated(with: p.weight_hh_l0, alongAxis: 1).transposed()
    let i = fusedWeight.shape[0]
    let j = fusedWeight.shape[1] / 4
    let fusedWeightTF = Tensor(
      concatenating: [
        fusedWeight.slice(lowerBounds: [0, 0], upperBounds: [i, j]),
        fusedWeight.slice(lowerBounds: [0, 2 * j], upperBounds: [i, 3 * j]),
        fusedWeight.slice(lowerBounds: [0, j], upperBounds: [i, 2 * j]),
        fusedWeight.slice(lowerBounds: [0, 3 * j], upperBounds: [i, 4 * j]),
      ],
      alongAxis: 1
    )

    let fusedBias = (p.bias_ih_l0 + p.bias_hh_l0)
    let k = fusedBias.shape[0] / 4
    let fusedBiasTF = Tensor(
      concatenating: [
        fusedBias.slice(lowerBounds: [0], upperBounds: [k]),
        fusedBias.slice(lowerBounds: [2 * k], upperBounds: [3 * k]),
        fusedBias.slice(lowerBounds: [k], upperBounds: [2 * k]),
        fusedBias.slice(lowerBounds: [3 * k], upperBounds: [4 * k]),
      ]
    )

    checkShapeAndSet(&lstm.fusedWeight, to: fusedWeightTF)
    checkShapeAndSet(&lstm.fusedBias, to: fusedBiasTF)
  }

  /// Sets the given MLP's parameters to the given MLP parameters.
  private func setMLP(_ mlp: inout MLP, to p: MLPParameters) {
    setDense(&mlp.dense1, to: p.linear1)
    setDense(&mlp.dense2, to: p.linear2)
  }

  /// Sets the given Dense's parameters to the given linear parameters.
  private func setDense(_ dense: inout Dense<Float>, to p: LinearParameters) {
    checkShapeAndSet(&dense.weight, to: p.weight.transposed())
    checkShapeAndSet(&dense.bias, to: p.bias)
  }
}

func tangentVector(from gradient: SNLMParameters, model: SNLM) -> SNLM.TangentVector {
  var model = model
  model.setParameters(gradient)

  // `model.setParameters` is for model parameters, not for gradients, so
  // we need to adjust the LSTM biases, whose gradients work differently than
  // `model.setParameters` does.
  model.encoderLSTM.cell.fusedBias /= 2
  model.decoderLSTM.cell.fusedBias /= 2

  return model.differentiableVectorView
}

func almostEqual(
  _ lhs: SNLM.TangentVector, _ rhs: SNLM.TangentVector, relTol: Float, zeroTol: Float
) -> Bool {
  var success = true
  for (kpIndex, kp) in lhs.recursivelyAllKeyPaths(to: Tensor<Float>.self).enumerated() {
    let t1 = lhs[keyPath: kp]
    let t2 = rhs[keyPath: kp]
    if t1.shape != t2.shape {
      print("Shape mismatch on tensor \(kpIndex)")
      success = false
      continue
    }

    if !(t1.isAlmostEqual(to: t2, tolerance: relTol)
      || (t1.isAlmostEqual(to: Tensor(zeros: t1.shape), tolerance: zeroTol)
        && t1.isAlmostEqual(to: Tensor(zeros: t1.shape), tolerance: zeroTol)))
    {
      print("Mismatch on tensor \(kpIndex)")
      success = false
    }
  }
  return success
}

class WordSegProbeLayerTests: XCTestCase {
  func testProbeEncoder() {
    // chrVocab is:
    // 0 - a
    // 1 - b
    // 2 - </s>
    // 3 - </w>
    // 4 - <pad>
    let chrVocab: Alphabet = Alphabet(
      [
        "a",
        "b",
      ], eos: "</s>", eow: "</w>", pad: "<pad>")

    // strVocab is:
    // 0 - aaaa
    // 1 - bbbb
    // 2 - abab
    let strVocab: Lexicon = Lexicon([
      CharacterSequence(alphabet: chrVocab, characters: [0, 0]),  // "aa"
      CharacterSequence(alphabet: chrVocab, characters: [1, 1]),  // "bb"
      CharacterSequence(alphabet: chrVocab, characters: [0, 1]),  // "ab"
      CharacterSequence(alphabet: chrVocab, characters: [1, 0]),  // "ba"
    ])

    var model = SNLM(
      parameters: SNLM.Parameters(
        ndim: 2,
        dropoutProb: 0,
        chrVocab: chrVocab,
        strVocab: strVocab,
        order: 5))

    model.setParameters(Example1.parameters)

    print("Encoding")
    let encoderStates = model.encode(
      CharacterSequence(alphabet: chrVocab, characters: [0, 1, 0, 1]))  // "abab"
    let encoderStatesTensor = Tensor(stacking: encoderStates)
    print("Expected: \(Example1.expectedEncoding)")
    print("Actual: \(encoderStatesTensor)")
    XCTAssert(abs(encoderStatesTensor - Example1.expectedEncoding).max().scalarized() < 1e-6)
    print("OK!\n")

    print("MLP Interpolation")
    let mlpInterpolationOutput = model.mlpInterpolation(encoderStatesTensor)
    print("Expected: \(Example1.expectedMLPInterpolationOutput)")
    print("Actual: \(encoderStates)")
    XCTAssert(
      abs(mlpInterpolationOutput - Example1.expectedMLPInterpolationOutput).max().scalarized()
        < 1e-6)
    print("OK!\n")

    print("MLP Memory")
    let mlpMemoryOutput = model.mlpMemory(encoderStatesTensor)
    print("Expected: \(Example1.expectedMLPMemoryOutput)")
    print("Actual: \(encoderStates)")
    XCTAssert(abs(mlpMemoryOutput - Example1.expectedMLPMemoryOutput).max().scalarized() < 1e-6)
    print("OK!\n")

    print("Decode")
    let decoded = model.decode(
      [
        CharacterSequence(alphabet: chrVocab, characters: [0, 0, 0]),  // "aaa"
        CharacterSequence(alphabet: chrVocab, characters: [0, 1]),  // "ab"
      ],
      encoderStates[0]
    )
    print("Expected: \(Example1.expectedDecoded)")
    print("Actual: \(decoded)")
    XCTAssert(abs(decoded - Example1.expectedDecoded).max().scalarized() < 1e-6)
    print("OK!\n")

    print("Build Lattice")
    let abab = CharacterSequence(alphabet: chrVocab, characters: [0, 1, 0, 1])
    let lattice = model.buildLattice(abab, maxLen: 5)
    XCTAssert(lattice.isAlmostEqual(to: Example1.lattice, tolerance: 1e-5))

    print("Gradient")
    func f(_ x: SNLM) -> Tensor<Float> {
      x.buildLattice(abab, maxLen: 5)[4].semiringScore.logr
    }
    let (_, grad) = valueWithGradient(at: model, in: f)
    let expectedGrad = tangentVector(from: Example1.gradWrtLogR, model: model)

    if !almostEqual(grad, expectedGrad, relTol: 1e-5, zeroTol: 1e-6) {
      print("\nExpected grad:\n\(expectedGrad)\n\n")
      print("Actual grad:\n\(grad)")
      XCTAssert(false, "Gradients wrong")
    }
  }

  static var allTests = [
    ("testProbeEncoder", testProbeEncoder)
  ]
}
