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

// Original source:
// "Big Transfer (BiT): General Visual Representation Learning"
// Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, Neil Houlsby.
// https://arxiv.org/abs/1912.11370

import Foundation
import TensorFlow
import PythonKit

let subprocess = Python.import("subprocess")
let np  = Python.import("numpy")


/// Convenient layer wrapper used to load all of the trained layers from the .npz file downloaded from the 
/// BigTransfer weights repository
struct BigTransferNamedLayer {
    let name: String
    let layer: Tensor<Float>
}

/// Get the necessary padding to maintain the network size specified in the BigTransfer architecture
///
/// - Parameters:
///   - kernelSize: size n which represents the height and width of the nxn kernel
/// - Returns: the left / top padding and the right / bottom padding necessary to maintain correct output sizes
///            after convolution
func paddingFromKernelSize(kernelSize: Int) -> [(before: Int, after: Int)] {
  let padTotal = kernelSize - 1
  let padBeginning = Int(padTotal / 2)
  let padEnd = padTotal - padBeginning
  let padding = [
        (before: 0, after: 0),
        (before: padBeginning, after: padEnd),
        (before: padBeginning, after: padEnd),
        (before: 0, after: 0)]
  return padding
}

/// Get all of the pre-trained layers from the .npz file into a Swift array to load into the BigTransfer model
///
/// - Parameters:
///   - modelName: model name that represents the weights to load from the BigTransfer weights repository 
///                ("BiT-M-R50x1" for example)
/// - Returns: an array of layers and their associated name in the .npz file downloaded from the weights repository
func getPretrainedWeightsDict(modelName: String) -> Array<BigTransferNamedLayer> {
  let validTypes = ["BiT-S", "BiT-M"]
  let validSizes = [(50, 1), (50, 3), (101, 1), (101, 3), (152, 4)]
  let bitURL = "https://storage.googleapis.com/bit_models/"
  var knownModels = [String: String]()

  for types in validTypes {
    for sizes in validSizes {
      let modelString = types + "-R" + String(sizes.0) + "x" + String(sizes.1)
      knownModels[modelString] = bitURL + modelString + ".npz"
    }
  }
  
  if let modelPath = knownModels[modelName] {
    subprocess.call("wget " + modelPath + " .", shell: true)
  }

  let weights = np.load("./" + modelName + ".npz")

  var weightsArray = Array<BigTransferNamedLayer>()
  for param in weights {
      weightsArray.append(BigTransferNamedLayer(name: String(param)!, layer: Tensor<Float>(numpy: weights[param])!))
  }
  return weightsArray
}

/// A 2D Convolution layer that standardizes the weights before the forward pass. This has been implemented in
/// accordance with the implementation in https://github.com/google-research/big_transfer/blob/49afe42338b62af9fbe18f0258197a33ee578a6b/bit_pytorch/models.py#L25
public struct StandardizedConv2D: Layer {
  public var conv: Conv2D<Float>

  public init(
    filterShape: (Int, Int, Int, Int),
    strides: (Int, Int) = (1, 1),
    padding: Padding = .valid,
    useBias: Bool = true
  )
  {
  self.conv = Conv2D(
      filterShape: filterShape, 
      strides: strides, 
      padding: padding,
      useBias: useBias)
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
      let axes: Array<Int> = [0, 1, 2]
      var standardizedConv = conv
      standardizedConv.filter = (standardizedConv.filter - standardizedConv.filter.mean(squeezingAxes: axes)) / sqrt((standardizedConv.filter.variance(squeezingAxes: axes) + 1e-16))
      return standardizedConv(input)
  }

}

/// A standardized convolution and group norm layer as specified in the BigTransfer architecture
public struct ConvGNV2BiT: Layer {
    public var conv: StandardizedConv2D
    public var norm: GroupNorm<Float>
    @noDerivative public var isSecond: Bool

    public init(
        inFilters: Int,
        outFilters: Int,
        kernelSize: Int = 1,
        stride: Int = 1,
        padding: Padding = .valid,
        isSecond: Bool = false
    ) {
        self.conv = StandardizedConv2D(
            filterShape: (kernelSize, kernelSize, inFilters, outFilters), 
            strides: (stride, stride), 
            padding: padding,
            useBias: false)
        self.norm = GroupNorm<Float>(
              offset: Tensor(zeros: [inFilters]),
              scale: Tensor(zeros: [inFilters]),
              groupCount: 2,
              axis: -1,
              epsilon: 0.001)
        self.isSecond = isSecond
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var normResult = norm(input)
        if self.isSecond {
            normResult = normResult.padded(forSizes: paddingFromKernelSize(kernelSize: 3))
        }
        let reluResult = relu(normResult)
        let convResult = conv(reluResult)
        return convResult
    }
}

/// The shortcut in a residual block with standardized convolution and group normalization
public struct ShortcutBiT: Layer {
    public var projection: StandardizedConv2D
    public var norm: GroupNorm<Float>
    @noDerivative public let needsProjection: Bool
    
    public init(inFilters: Int, outFilters: Int, stride: Int) {
      needsProjection = (stride > 1 || inFilters != outFilters)
      norm = GroupNorm<Float>(
          offset: Tensor(zeros: [needsProjection ? inFilters  : 1]),
          scale: Tensor(zeros: [needsProjection ? inFilters  : 1]),
          groupCount: needsProjection ? 2  : 1,
          axis: -1,
          epsilon: 0.001)
        
        projection =  StandardizedConv2D(
            filterShape: (1, 1, needsProjection ? inFilters  : 1, needsProjection ? outFilters : 1), 
            strides: (stride, stride), 
            padding: .valid,
            useBias: false)
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var res = input
        if needsProjection { 
          res = norm(res)
          res = relu(res)
          res = projection(res)
        }
        return res
    }
}

/// Residual block for BigTransfer with standardized convolution and group normalization layers
public struct ResidualBlockBiT: Layer {
    public var shortcut: ShortcutBiT
    public var convs: [ConvGNV2BiT]

    public init(inFilters: Int, outFilters: Int, stride: Int, expansion: Int){
        if expansion == 1 {
            convs = [
                ConvGNV2BiT(inFilters: inFilters,  outFilters: outFilters, kernelSize: 3, stride: stride),
                ConvGNV2BiT(inFilters: outFilters, outFilters: outFilters, kernelSize: 3, isSecond: true)
            ]
        } else {
            convs = [
                ConvGNV2BiT(inFilters: inFilters,    outFilters: outFilters/4),
                ConvGNV2BiT(inFilters: outFilters/4, outFilters: outFilters/4, kernelSize: 3, stride: stride, isSecond: true),
                ConvGNV2BiT(inFilters: outFilters/4, outFilters: outFilters)
            ]
        }
        shortcut = ShortcutBiT(inFilters: inFilters, outFilters: outFilters, stride: stride)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let convResult = convs.differentiableReduce(input) { $1($0) }
        return convResult + shortcut(input)
    }
}

/// An implementation of the BigTransfer architecture with variable sizes
public struct BigTransfer: Layer {
  public var inputStem: StandardizedConv2D
  public var maxPool: MaxPool2D<Float>
  public var residualBlocks: [ResidualBlockBiT] = []
  public var groupNorm : GroupNorm<Float>
  public var flatten = Flatten<Float>()
  public var classifier: Dense<Float>
  public var avgPool = GlobalAvgPool2D<Float>()
  @noDerivative public var finalOutFilter : Int = 0

  /// Initialize the BigTransfer Model
  ///
  /// - Parameters:
  ///   - classCount: the number of output classes
  ///   - depth: the specified depht of the network based on the various ResNet architectures
  ///   - inputChannels: the number of input channels for the dataset
  ///   - stemFilters: the number of filters in the first three convolutions
  public init(
        classCount: Int, 
        depth: Depth, 
        inputChannels: Int = 3,
        modelName: String = "BiT-M-R50x1",
        loadWeights: Bool = true
    ) {

        self.inputStem = StandardizedConv2D(filterShape: (7, 7, 3, 64), strides: (2, 2), padding: .valid, useBias: false)
        self.maxPool = MaxPool2D(poolSize: (3, 3), strides: (2, 2), padding: .valid)
        let sizes = [64 / depth.expansion, 64, 128, 256, 512]
        for (iBlock, nBlocks) in depth.layerBlockSizes.enumerated() {
            let (nIn, nOut) = (sizes[iBlock] * depth.expansion, sizes[iBlock+1] * depth.expansion)
            for j in 0..<nBlocks {

                self.residualBlocks.append(ResidualBlockBiT(
                    inFilters: j==0 ? nIn : nOut,  
                    outFilters: nOut, 
                    stride: (iBlock != 0) && (j == 0) ? 2 : 1, 
                    expansion: depth.expansion
                ))
                self.finalOutFilter = nOut
            }
        }
        self.groupNorm = GroupNorm<Float>(
              offset: Tensor(zeros: [self.finalOutFilter]),
              scale: Tensor(zeros: [self.finalOutFilter]),
              groupCount: 2,
              axis: -1,
              epsilon: 0.001)
        self.classifier = Dense(inputSize: 512 * depth.expansion, outputSize: classCount)
        
        if loadWeights {
            let weightsArray = getPretrainedWeightsDict(modelName: modelName)

            // Load weights from model .npz file into the BigTransfer model
            let convs = weightsArray.filter {key in return key.name.contains("/block") && key.name.contains("standardized_conv2d/kernel") && !(key.name.contains("proj"))}
            
            var k = 0
            for (idx, i) in self.residualBlocks.enumerated() {
                for (jdx, _) in i.convs.enumerated() {
                assert(self.residualBlocks[idx].convs[jdx].conv.conv.filter.shape == convs[k].layer.shape)
                self.residualBlocks[idx].convs[jdx].conv.conv.filter = convs[k].layer
                k = k + 1
                }
            }

            let projectiveConvs = weightsArray.filter {key in return key.name.contains("/block") && key.name.contains("standardized_conv2d/kernel") && (key.name.contains("proj"))}
            var normScale = weightsArray.filter {key in return key.name.contains("unit01/a/group_norm/gamma")}
            var normOffset = weightsArray.filter {key in return key.name.contains("unit01/a/group_norm/beta")}

            k = 0
            for (idx, i) in self.residualBlocks.enumerated() {
                if (i.shortcut.projection.conv.filter.shape != [1, 1, 1, 1])
                {
                    assert(self.residualBlocks[idx].shortcut.projection.conv.filter.shape == projectiveConvs[k].layer.shape)
                    self.residualBlocks[idx].shortcut.projection.conv.filter = projectiveConvs[k].layer

                    assert(self.residualBlocks[idx].shortcut.norm.scale.shape == normScale[k].layer.shape)
                    self.residualBlocks[idx].shortcut.norm.scale = normScale[k].layer

                    assert(self.residualBlocks[idx].shortcut.norm.offset.shape == normOffset[k].layer.shape)
                    self.residualBlocks[idx].shortcut.norm.offset = normOffset[k].layer
                    k = k + 1
                }
            }

            normScale = weightsArray.filter {key in return key.name.contains("gamma")}

            k = 0
            for (idx, i) in self.residualBlocks.enumerated() {
                for (jdx, _) in i.convs.enumerated() {
                assert(normScale[k].layer.shape == self.residualBlocks[idx].convs[jdx].norm.scale.shape)
                self.residualBlocks[idx].convs[jdx].norm.scale = normScale[k].layer
                k = k + 1
                }
            }

            normOffset = weightsArray.filter {key in return key.name.contains("beta")}

            var l = 0
            for (idx, i) in self.residualBlocks.enumerated() {
                for (jdx, _) in i.convs.enumerated() {
                assert(normOffset[l].layer.shape == self.residualBlocks[idx].convs[jdx].norm.offset.shape)
                self.residualBlocks[idx].convs[jdx].norm.offset = normOffset[l].layer
                l = l + 1
                }
            }

            assert(self.groupNorm.scale.shape == normScale[k].layer.shape)
            self.groupNorm.scale = normScale[k].layer
            assert(self.groupNorm.offset.shape == normOffset[l].layer.shape)
            self.groupNorm.offset = normOffset[l].layer

            let rootConvs = weightsArray.filter {key in return key.name.contains("root_block")}
            assert(self.inputStem.conv.filter.shape == rootConvs[0].layer.shape)
            self.inputStem.conv.filter = rootConvs[0].layer
        }
    }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
      var paddedInput = input.padded(forSizes: paddingFromKernelSize(kernelSize: 7))
      paddedInput = inputStem(paddedInput).padded(forSizes: paddingFromKernelSize(kernelSize: 3))
      let inputLayer = maxPool(paddedInput)
      let blocksReduced = residualBlocks.differentiableReduce(inputLayer) { $1($0) }
      let normalized = relu(groupNorm(blocksReduced))
      return normalized.sequenced(through: avgPool, flatten, classifier)
  }
}

extension BigTransfer {
    public enum Depth {
        case resNet18
        case resNet34
        case resNet50
        case resNet101
        case resNet152

        var expansion: Int {
            switch self {
            case .resNet18, .resNet34: return 1
            default: return 4
            }
        }

        var layerBlockSizes: [Int] {
            switch self {
            case .resNet18:  return [2, 2, 2,  2]
            case .resNet34:  return [3, 4, 6,  3]
            case .resNet50:  return [3, 4, 6,  3]
            case .resNet101: return [3, 4, 23, 3]
            case .resNet152: return [3, 8, 36, 3]
            }
        }
    }
}
