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

struct WavenetModel: Layer {
    public typealias Input = Tensor<Float>
    public typealias Output = Tensor<Float>

    // Parameters
    @noDerivative
    let batchSize: Int
    @noDerivative
    let dilations: [Int]
    @noDerivative
    let filterWidth: Int
    @noDerivative
    let residualChannels: Int
    @noDerivative
    let dilationChannels: Int
    @noDerivative
    let skipChannels: Int
    @noDerivative
    let quantizationChannels: Int
    @noDerivative
    let useBiases: Bool
    @noDerivative
    let receptiveFieldWidth: Int
    @noDerivative
    let l2RegularizationWeight: Float

    var causalConv1: CausalConvLayer
    var dilationLayers: [DilationLayer]
    var postprocessingConv1: Conv1D<Float>
    var postprocessingConv2: Conv1D<Float>
    var postprocessingBias1: Tensor<Float>
    var postprocessingBias2: Tensor<Float>

    public init(
        inputChannels _: Int,
        outputChannels _: Int,
        batchSize: Int,
        dilations: [Int],
        initialInputWidth: Int,
        filterWidth: Int,
        residualChannels: Int,
        dilationChannels: Int,
        skipChannels: Int,
        quantizationChannels: Int = 256,
        useBiases: Bool = true,
        l2RegularizationWeight: Float = 0.0
    ) {
        self.batchSize = batchSize
        self.dilations = dilations
        self.filterWidth = filterWidth
        self.residualChannels = residualChannels
        self.dilationChannels = dilationChannels
        self.skipChannels = skipChannels
        self.quantizationChannels = quantizationChannels
        self.useBiases = useBiases
        self.l2RegularizationWeight = l2RegularizationWeight

        receptiveFieldWidth = WavenetModel.computeReceptiveFieldWidth(
            filterWidth: filterWidth,
            dilations: dilations
        )
        causalConv1 = CausalConvLayer(
            inChannels: quantizationChannels,
            outChannels: residualChannels,
            kernelSize: filterWidth,
            dilation: 1
        )
        dilationLayers = []
        for dilation in dilations {
            dilationLayers.append(
                DilationLayer(
                    dilation: dilation,
                    filterWidth: filterWidth,
                    initialInputWidth: initialInputWidth,
                    inChannels: residualChannels,
                    outChannels: dilationChannels,
                    skipChannels: skipChannels,
                    receptiveFieldWidth: receptiveFieldWidth,
                    useBiases: useBiases
                )
            )
        }
        postprocessingConv1 = Conv1D(
            filterShape: (1, skipChannels, skipChannels),
            padding: Padding.same
        )
        postprocessingConv2 = Conv1D(
            filterShape: (1, skipChannels, quantizationChannels),
            padding: Padding.same
        )
        postprocessingBias1 = Tensor<Float>(
            repeating: 0.0,
            shape: [skipChannels]
        )
        postprocessingBias2 = Tensor<Float>(
            repeating: 0.0,
            shape: [quantizationChannels]
        )
    }

    public static func computeReceptiveFieldWidth(
        filterWidth: Int,
        dilations: [Int]
    ) -> Int {
        var receptiveField = (filterWidth - 1) * dilations.reduce(0, +) + 1
        receptiveField += filterWidth - 1
        return receptiveField
    }

    @differentiable
    public func callAsFunction(_ input: Input) -> Output {
        var currLayer = causalConv1(input)
        var skipOutputs: [Output] = []
        for (index, _) in dilations.enumerated() {
            let layerOutput = dilationLayers[index](currLayer)
            currLayer = layerOutput.layerOutput
            skipOutputs.append(layerOutput.skipOutput)
        }
        let output = skipOutputs.differentiableReduce(
            Tensor<Float>(repeating: 0.0, shape: skipOutputs[0].shape), +
        )
        var postprocessingOutput = postprocessingConv1(
            relu(output))
        if useBiases {
            postprocessingOutput = postprocessingOutput + postprocessingBias1
        }
        postprocessingOutput = postprocessingConv2(relu(postprocessingOutput))
        if useBiases {
            postprocessingOutput = postprocessingOutput + postprocessingBias2
        }
        return postprocessingOutput
    }

    public func computeInputsAndLabels(_ input: Input)
        -> (Input, Tensor<Float32>) {
        let encodedInput = muLawEncode(input,
                                       quantizationChannels: quantizationChannels)
        let encoded = Tensor<Float>(oneHotAtIndices: encodedInput,
                                    depth: quantizationChannels)
        var networkInput: Input
        networkInput = encoded

        // Cut off the last sample of network input to preserve causality.
        let networkInputWidth = networkInput.shape[1] - 1
        networkInput = networkInput.slice(
            lowerBounds: Tensor<Int32>([0, 0, 0]),
            sizes: Tensor<Int32>([-1, Int32(networkInputWidth), -1])
        )

        // TODO(akshaan): Reconsider the 2*receptiveField
        // window size once the PaddingFIFOQueue is being used
        // to hold the network input tensors
        let labels = encoded.slice(
            lowerBounds: Tensor<Int32>([0, Int32(2 * receptiveFieldWidth - 1), 0]),
            sizes: Tensor<Int32>([-1, -1, -1])
        )

        return (networkInput, labels)
    }

    @differentiable(wrt: predictions)
    public func loss(predictions: Output, labels: Tensor<Float32>) -> Tensor<Float> {
        let flattenedPredictions = predictions.reshaped(
            toShape: Tensor<Int32>([-1, Int32(quantizationChannels)]))
        let flattenedLabels = labels.reshaped(to: [-1, quantizationChannels])
        return softmaxCrossEntropy(
            logits: flattenedPredictions,
            probabilities: flattenedLabels
        ).mean()
    }
}
