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

public class PythonCheckpointReader {
    private let path: String
    private var layerCounts: [String: Int] = [:]

    public init(path: String) {
        self.path = path
    }

    // Currently returns an `Optional` in order to support the case where the variable might not
    // exist, but this is not implemented (see b/124126672).
    func readTensor(layerName: String, weightName: String) -> Tensor<Float>? {
        let countSuffix = layerCounts[layerName] == nil ? "" : "_\(layerCounts[layerName]!)"
        let tensorName = layerName + countSuffix + "/" + weightName
        // TODO(jekbradbury): support variadic dtype attrs in RawOpsGenerated
        return _Raw.restoreV2(prefix: StringTensor(path),
                             tensorNames: StringTensor([tensorName]),
                             shapeAndSlices: StringTensor([""]))
    }

    /// Increments a per-layer counter for variable names in the checkpoint file.
    /// As the Python model code uses low-level TensorFlow APIs, variables are namespaced only by
    /// layer name and this per-layer counter (e.g., conv2d_5/bias).
    func increment(layerName: String) {
        layerCounts[layerName, default: 0] += 1
    }
}

private func checkShapes(_ tensor1: Tensor<Float>, _ tensor2: Tensor<Float>) {
    guard tensor1.shape == tensor2.shape else {
        print("Shape mismatch: \(tensor1.shape) != \(tensor2.shape)")
        fatalError()
    }
}

protocol LoadableFromPythonCheckpoint {
    mutating func load(from reader: PythonCheckpointReader)
}

extension Dense: LoadableFromPythonCheckpoint where Scalar == Float {
    mutating func load(from reader: PythonCheckpointReader) {
        let newWeight = reader.readTensor(layerName: "dense", weightName: "kernel")!
        checkShapes(weight, newWeight)
        weight = newWeight

        if let newBias = reader.readTensor(layerName: "dense", weightName: "bias") {
            checkShapes(bias, newBias)
            bias = newBias
        }
        reader.increment(layerName: "dense")
    }
}

extension Conv2D: LoadableFromPythonCheckpoint where Scalar == Float {
    mutating func load(from reader: PythonCheckpointReader) {
        let newFilter = reader.readTensor(layerName: "conv2d", weightName: "kernel")!
        checkShapes(filter, newFilter)
        filter = newFilter

        // TODO(jekbradbury): handle layers with optional weights
        // It would be helpful to have an op to see if a checkpoint contains a particular variable
        // (see b/124126672)
        // if let newBias = loader.readTensor(layerName: "conv2d", weightName: "bias") {
        //   checkShapes(bias, newBias)
        //   bias = newBias
        // }

        reader.increment(layerName: "conv2d")
    }
}

extension BatchNorm: LoadableFromPythonCheckpoint where Scalar == Float {
    mutating func load(from reader: PythonCheckpointReader) {
        if let newOffset = reader.readTensor(layerName: "batch_normalization", weightName: "beta") {
            checkShapes(offset, newOffset)
            offset = newOffset
        }

        if let newScale = reader.readTensor(layerName: "batch_normalization", weightName: "gamma") {
            checkShapes(scale, newScale)
            scale = newScale
        }

        if let newRunningMean = reader.readTensor(
            layerName: "batch_normalization",
            weightName: "moving_mean") {
            // Do not check shapes, because Swift running mean/variance are initialized to scalar
            // tensors.
            runningMean.value = newRunningMean
        }

        if let newRunningVariance = reader.readTensor(
            layerName: "batch_normalization",
            weightName: "moving_variance") {
            // Do not check shapes, because Swift running mean/variance are initialized to scalar
            // tensors.
            runningVariance.value = newRunningVariance
        }

        reader.increment(layerName: "batch_normalization")
    }
}
