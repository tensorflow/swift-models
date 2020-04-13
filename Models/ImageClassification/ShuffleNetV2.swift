// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

import TensorFlow

// Original V2 paper
// "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
// Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun

public struct ChannelShuffle: ParameterlessLayer {
    @noDerivative public var groups: Int
    
    public init(groups: Int = 2) {
        self.groups = groups
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let batchSize = input.shape[0], height = input.shape[1], width = input.shape[2],
        channels = input.shape[3]
        let channelsPerGroup: Int = channels / groups
        
        var output = input.reshaped(to: [batchSize, height, width, groups, channelsPerGroup])
        output = output.transposed(permutation: [0, 1, 2, 4, 3])
        output = output.reshaped(to: [batchSize, height, width, channels])
        return output
    }
}

public struct InvertedResidual: Layer {
    @noDerivative public var includeBranch: Bool = true
    @noDerivative public var zeropad: ZeroPadding2D = ZeroPadding2D<Float>(padding: ((1, 1), (1, 1)))
    
    public var branch: Sequential<ZeroPadding2D<Float>, Sequential<DepthwiseConv2D<Float>,
    Sequential<BatchNorm<Float>, Sequential<Conv2D<Float>, BatchNorm<Float>>>>>
    public var conv1: Conv2D<Float>
    public var batchNorm1: BatchNorm<Float>
    public var depthwiseConv: DepthwiseConv2D<Float>
    public var batchNorm2: BatchNorm<Float>
    public var conv2: Conv2D<Float>
    public var batchNorm3: BatchNorm<Float>
    
    public init(filters: (Int, Int), stride: Int) {
        if stride == 1 {
            includeBranch = false
        }
        
        let branchChannels = filters.1 / 2
        branch = Sequential {
            ZeroPadding2D<Float>(padding: ((1, 1), (1, 1)))
            DepthwiseConv2D<Float>(
                filterShape: (3, 3, filters.0, 1), strides: (stride, stride),
                padding: .valid
            )
            BatchNorm<Float>(featureCount: filters.0)
            Conv2D<Float>(
                filterShape: (1, 1, filters.0, branchChannels), strides: (1, 1), padding: .valid,
                useBias: false
            )
            BatchNorm<Float>(featureCount: branchChannels)
        }
        var inputChannels = includeBranch ? filters.0: branchChannels
        conv1 = Conv2D<Float>(
            filterShape: (1, 1, inputChannels, branchChannels), strides: (1, 1), padding: .valid,
            useBias: false
        )
        conv2 = Conv2D<Float>(
            filterShape: (1, 1, branchChannels, branchChannels), strides: (1, 1), padding: .valid,
            useBias: false
        )
        depthwiseConv = DepthwiseConv2D<Float>(
            filterShape: (3, 3, branchChannels, 1), strides: (stride, stride), padding: .valid
        )
        batchNorm1 = BatchNorm<Float>(featureCount: branchChannels)
        batchNorm2 = BatchNorm<Float>(featureCount: branchChannels)
        batchNorm3 = BatchNorm<Float>(featureCount: branchChannels)
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        if !includeBranch {
            let splitInput = input.split(count: 2, alongAxis: 3)
            let input1 = splitInput[0]
            let input2 = splitInput[1]
            var output2 = relu(input2.sequenced(through: conv1, batchNorm1))
            output2 = relu(output2.sequenced(through: zeropad, depthwiseConv, batchNorm2, conv2,
                                             batchNorm3))
            return ChannelShuffle()(input1.concatenated(with: output2, alongAxis: 3))
        } else {
            let output1 = branch(input)
            var output2 = relu(input.sequenced(through: conv1, batchNorm1))
            output2 = relu(output2.sequenced(through: zeropad, depthwiseConv, batchNorm2, conv2,
                                             batchNorm3))
            return ChannelShuffle()(output1.concatenated(with: output2, alongAxis: 3))
        }
    }
}



public struct ShuffleNetV2: Layer {
    @noDerivative public var zeroPad: ZeroPadding2D<Float> = ZeroPadding2D<Float>(padding: ((1, 1), (1, 1)))
    
    public var conv1: Conv2D<Float>
    public var batchNorm1: BatchNorm<Float>
    public var maxPool: MaxPool2D<Float>
    public var invertedResidualBlocksStage1: [InvertedResidual]
    public var invertedResidualBlocksStage2: [InvertedResidual]
    public var invertedResidualBlocksStage3: [InvertedResidual]
    public var conv2: Conv2D<Float>
    public var globalPool: GlobalAvgPool2D<Float> = GlobalAvgPool2D()
    public var dense: Dense<Float>
    
    public init(stagesRepeat: (Int, Int, Int), stagesOutputChannels: (Int, Int, Int, Int, Int),
                classCount: Int) {
        var inputChannels = 3
        var outputChannels = stagesOutputChannels.0
        conv1 = Conv2D<Float>(
            filterShape: (3, 3, inputChannels, outputChannels), strides: (1, 1)
        )
        maxPool = MaxPool2D(poolSize: (3, 3), strides: (2, 2))
        conv2 = Conv2D<Float>(
            filterShape: (1, 1, stagesOutputChannels.3, stagesOutputChannels.4), strides: (1, 1),
            useBias: false
        )
        dense = Dense<Float>(inputSize: stagesOutputChannels.4, outputSize: classCount)
        batchNorm1 = BatchNorm(featureCount: outputChannels)
        inputChannels = outputChannels
        outputChannels = stagesOutputChannels.1
        invertedResidualBlocksStage1 = [InvertedResidual(filters: (inputChannels, outputChannels),
                                                         stride: 2)]
        for _ in 1...stagesRepeat.0 {
            invertedResidualBlocksStage1.append(InvertedResidual(
                filters: (outputChannels, outputChannels), stride: 1)
            )
        }
        inputChannels = outputChannels
        outputChannels = stagesOutputChannels.2
        invertedResidualBlocksStage2 = [InvertedResidual(filters: (inputChannels, outputChannels),
                                                         stride: 2)]
        for _ in 1...stagesRepeat.1 {
            invertedResidualBlocksStage2.append(InvertedResidual(
                filters: (outputChannels, outputChannels), stride: 1)
            )
        }
        
        inputChannels = outputChannels
        outputChannels = stagesOutputChannels.3
        invertedResidualBlocksStage3 = [InvertedResidual(filters: (inputChannels, outputChannels),
                                                         stride: 2)]
        for _ in 1...stagesRepeat.2 {
            invertedResidualBlocksStage3.append(InvertedResidual(
                filters: (outputChannels, outputChannels), stride: 1)
            )
        }
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var output = relu(input.sequenced(through: zeroPad, conv1, batchNorm1, zeroPad, maxPool))
        output = invertedResidualBlocksStage1.differentiableReduce(output) {$1($0)}
        output = invertedResidualBlocksStage2.differentiableReduce(output) {$1($0)}
        output = invertedResidualBlocksStage3.differentiableReduce(output) {$1($0)}
        output = relu(conv2(output))
        return output.sequenced(through: globalPool, dense)
    }
}

extension ShuffleNetV2 {
    public enum Kind {
        case shuffleNetV2x05
        case shuffleNetV2x10
        case shuffleNetV2x15
        case shuffleNetV2x20
    }

    public init(kind: Kind) {
        switch kind {
        case .shuffleNetV2x05:
            self.init(
                stagesRepeat: (4, 8, 4), stagesOutputChannels: (24, 48, 96, 192, 1024),
                classCount: 1000
            )
        case .shuffleNetV2x10:
            self.init(
                stagesRepeat: (4, 8, 4), stagesOutputChannels: (24, 116, 232, 464, 1024),
                classCount: 1000
            )
        case .shuffleNetV2x15:
            self.init(
                stagesRepeat: (4, 8, 4), stagesOutputChannels: (24, 176, 352, 704, 1024),
                classCount: 1000
            )
        case .shuffleNetV2x20:
            self.init(
                stagesRepeat: (4, 8, 4), stagesOutputChannels: (24, 244, 488, 976, 2048),
                classCount: 1000
            )
        }
    }
}
