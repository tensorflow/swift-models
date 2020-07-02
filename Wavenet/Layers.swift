import Foundation
// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing software
// distributed under the License is distributed on an "AS IS" BASIS
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import TensorFlow

/// Perform mu-law companding transformation (ITU-T, 1988).
/// Minimum operation is here to deal with rare large amplitudes caused
/// by resampling.
///
/// - Parameter input: The input to the layer.
/// - Parameter quantizationChannels: Num. quantization channels to use
/// - Returns: The output.
public func muLawEncode(_ input: Tensor<Float>, quantizationChannels: Int) -> Tensor<Int32> {
    let mu = Float(quantizationChannels - 1)
    let safe_audio_abs = min(abs(input), 1.0)
    let magnitude = log1p(mu * safe_audio_abs) / log1p(mu)
    let signal = sign(input) * magnitude
    return Tensor<Int32>((signal + 1) / 2 * mu + 0.5)
}

/// Perform reverse mu-law transformation.
///
/// - Parameter input: The input to the layer.
/// - Parameter quantizationChannels: Num. quantization channels to use
/// - Returns: The output.
public func muLawDecode(_ input: Tensor<Int32>, quantizationChannels: Int) -> Tensor<Float> {
    let mu = Float(quantizationChannels - 1)
    let signal = 2.0 * (Tensor<Float32>(input) / mu) - 1
    let magnitude = (1 / mu) * pow(1 + mu, abs(signal) - 1)
    return sign(signal) * magnitude
}

public struct CausalConvLayer: Layer {
    public typealias Input = Tensor<Float>
    public typealias Output = Tensor<Float>

    /// Convolution layer.
    public var conv1d: Conv1D<Float>

    /// Dilation rate
    @noDerivative
    public var dilation: Int

    /// Kernel size
    @noDerivative
    public var kernelSize: Int

    /// Creates a 1D causal convolution with padding layer.
    ///
    /// - Parameters:
    ///   - inChannels: Number of input channels in convolution kernel.
    ///   - outChannels: Number of output channels in convolution kernel.
    ///   - kernelSize: Convolution kernel size (both width and height).
    public init(inChannels: Int, outChannels: Int, kernelSize: Int, dilation: Int) {
        conv1d = Conv1D(filterShape: (kernelSize, inChannels, outChannels))
        self.dilation = dilation
        self.kernelSize = kernelSize
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func callAsFunction(_ input: Input) -> Output {
        let outWidth = input.shape[1] - (kernelSize - 1) * dilation
        if dilation > 1 {
            let transformed = timeToBatch(input)
            let convolved = conv1d(transformed)
            let restored = batchToTime(convolved)
            return restored.slice(
                lowerBounds: [0, 0, 0],
                sizes: Tensor<Int32>([-1, Int32(outWidth), -1])
            )
        } else {
            let convolved = conv1d(input)
            return convolved.slice(
                lowerBounds: [0, 0, 0],
                sizes: Tensor<Int32>([-1, Int32(outWidth), -1])
            )
        }
    }

    private func batchToTime(_ input: Input) -> Output {
        let prepared = input.reshaped(to: [dilation, -1, input.shape[2]])
        let transposed = prepared.transposed(permutation: [1, 0, 2])
        return transposed.reshaped(to: [input.shape[0] / dilation, -1, input.shape[2]])
    }

    private func timeToBatch(_ input: Input) -> Output {
        let shape = input.shape
        let padElements = dilation - 1 - (shape[1] + dilation - 1) % dilation
        let padded = input.padded(forSizes: [(0, 0), (0, padElements), (0, 0)])
        let reshaped = padded.reshaped(to: [-1, dilation, shape[2]])
        let transposed = reshaped.transposed(permutation: [1, 0, 2])
        return transposed.reshaped(to: [shape[0] * dilation, -1, shape[2]])
    }
}

public struct SkipOutput: Differentiable {
    var skipOutput: Tensor<Float>
    var layerOutput: Tensor<Float>
}

public struct DilationLayer: Layer {
    public typealias Input = Tensor<Float>
    public typealias Output = SkipOutput

    var convFilter: CausalConvLayer
    var convGate: CausalConvLayer
    var filterBias: Tensor<Float>
    var gateBias: Tensor<Float>
    var transformer: Conv1D<Float>
    var denseBias: Tensor<Float>
    var skipContribution: Conv1D<Float>
    var skipBias: Tensor<Float>

    @noDerivative
    var receptiveFieldWidth: Int

    @noDerivative
    var useBiases: Bool

    @noDerivative
    var initialInputWidth: Int

    public init(dilation: Int,
                filterWidth: Int,
                initialInputWidth: Int,
                inChannels: Int,
                outChannels: Int,
                skipChannels: Int,
                receptiveFieldWidth: Int,
                useBiases: Bool) {
        convFilter = CausalConvLayer(inChannels: inChannels, outChannels: outChannels, kernelSize: filterWidth, dilation: dilation)
        convGate = CausalConvLayer(inChannels: inChannels, outChannels: outChannels, kernelSize: filterWidth, dilation: dilation)

        filterBias = Tensor<Float>(repeating: 0.0, shape: [outChannels])
        gateBias = Tensor<Float>(repeating: 0.0, shape: [outChannels])

        denseBias = Tensor<Float>(repeating: 0.0,
                                  shape: [inChannels])

        transformer = Conv1D(filterShape: (1, outChannels, inChannels), stride: 1, padding: Padding.same)

        skipBias = Tensor<Float>(repeating: 0.0,
                                 shape: [skipChannels])

        skipContribution = Conv1D(filterShape: (1, outChannels, skipChannels), padding: Padding.same)

        self.useBiases = useBiases
        self.receptiveFieldWidth = receptiveFieldWidth
        self.initialInputWidth = initialInputWidth
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> SkipOutput {
        var convFilterOut = convFilter(input)
        var convGateOut = convGate(input)

        if useBiases {
            convFilterOut = convFilterOut + filterBias
            convGateOut = convGateOut + gateBias
        }

        let filterAndGateOut = tanh(convFilterOut) + sigmoid(convGateOut)
        var transformerOut = transformer(filterAndGateOut)
        if useBiases {
            transformerOut = transformerOut + denseBias
        }

        let inputCut = Int32(filterAndGateOut.shape[1] - transformerOut.shape[1])
        let inputSlice = filterAndGateOut.slice(
            lowerBounds: Tensor<Int32>([0, Int32(inputCut), 0]),
            sizes: Tensor<Int32>([-1, -1, -1])
        )
        transformerOut = transformerOut + inputSlice

        // The 1x1 conv to produce the skip output
        let skipCut = filterAndGateOut.shape[1] - (initialInputWidth - receptiveFieldWidth + 1)
        let outSkip = filterAndGateOut.slice(
            lowerBounds: Tensor<Int32>([0, Int32(skipCut), 0]),
            sizes: Tensor<Int32>([-1, -1, -1])
        )
        var skipContributionOut = skipContribution(outSkip)

        if useBiases {
            skipContributionOut = skipContributionOut + skipBias
        }

        return SkipOutput(skipOutput: skipContributionOut, layerOutput: transformerOut)
    }
}
