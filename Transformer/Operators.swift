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

// Contains operators needed for Transformer; these will likely be upstreamed into swift-apis

/// Computes the Gaussian error linear unit (GELU) nonlinear activation function
@differentiable
func gelu<Scalar: TensorFlowFloatingPoint>(_ x: Tensor<Scalar>) -> Tensor<Scalar> {
    let xCubed = x * x * x
    let polynomial = 0.79788456 * (x + 0.044715 * xCubed)
    return 0.5 * x * (1.0 + tanh(polynomial))
}

/// Performs batched matrix multiplication of two tensors. The last two axes of each tensor
/// are treated as the matrix axes; all others are treated as batch axes.
@differentiable(
    wrt: (left, right),
    vjp: _vjpBatchedMatmul
    where Scalar : Differentiable & FloatingPoint
)
func batchedMatmul<Scalar : Numeric>(
    _ left: Tensor<Scalar>,
    _ right: Tensor<Scalar>,
    adjointLeft: Bool = false,
    adjointRight: Bool = false
) -> Tensor<Scalar> {
    return Raw.batchMatMul(left, right, adjX: adjointLeft, adjY: adjointRight)
}

@usableFromInline
func _vjpBatchedMatmul<Scalar : Differentiable & FloatingPoint>(
    _ left: Tensor<Scalar>,
    _ right: Tensor<Scalar>,
    adjointLeft: Bool,
    adjointRight: Bool
) -> (Tensor<Scalar>, (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
    let value = batchedMatmul(left, right, adjointLeft: adjointLeft, adjointRight: adjointRight)
    return (value, { v in
        if !adjointLeft {
            if !adjointRight {
                return (
                    batchedMatmul(v, right, adjointLeft: false, adjointRight: true),
                    batchedMatmul(left, v, adjointLeft: true, adjointRight: false))
            } else {
                return (
                    batchedMatmul(v, right, adjointLeft: false, adjointRight: false),
                    batchedMatmul(v, left, adjointLeft: true, adjointRight: false))
            }
        } else {
            if !adjointRight {
                return (
                    batchedMatmul(right, v, adjointLeft: false, adjointRight: true),
                    batchedMatmul(left, v, adjointLeft: false, adjointRight: false))
            } else {
                return (
                    batchedMatmul(right, v, adjointLeft: true, adjointRight: true),
                    batchedMatmul(v, left, adjointLeft: true, adjointRight: true))
            }
        }
    })
}
