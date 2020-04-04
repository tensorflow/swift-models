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

import TensorFlow

/// MLP is a multi-layer perceptron and is used as a component of the DLRM model
public struct MLP: Layer {
    public var blocks: [Dense<Float>] = []

    /// Randomly initializes a new multilayer perceptron from the given hyperparameters.
    ///
    /// - Parameter dims: Dims represents the size of the input, hidden layers, and output of the
    ///   multi-layer perceptron.
    /// - Parameter sigmoidLastLayer: if `true`, use a `sigmoid` activation function for the last layer,
    ///   `relu` otherwise.
    init(dims: [Int], sigmoidLastLayer: Bool = false) {
        for i in 0..<(dims.count-1) {
            if sigmoidLastLayer && i == dims.count - 2 {
                blocks.append(Dense(inputSize: dims[i], outputSize: dims[i+1], activation: sigmoid))
            } else {
                blocks.append(Dense(inputSize: dims[i], outputSize: dims[i+1], activation: relu))
            }
        }
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let blocksReduced = blocks.differentiableReduce(input) { last, layer in
            layer(last)
        }
        return blocksReduced
    }

}

