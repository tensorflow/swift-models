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

/// Returns a function that creates a tensor by initializing all its values randomly from a
/// truncated Normal distribution. The generated values follow a Normal distribution with mean
/// `mean` and standard deviation `standardDeviation`, except that values whose magnitude is more
/// than two standard deviations from the mean are dropped and resampled.
///
/// - Parameters:
///   - mean: Mean of the Normal distribution.
///   - standardDeviation: Standard deviation of the Normal distribution.
///
///- Returns: A truncated normal parameter initializer function.
public func truncatedNormalInitializer<Scalar: TensorFlowFloatingPoint>(
    mean: Tensor<Scalar> = Tensor<Scalar>(0),
    standardDeviation: Tensor<Scalar> = Tensor<Scalar>(1),
    seed: TensorFlowSeed = TensorFlow.Context.local.randomSeed
) -> ParameterInitializer<Scalar> {
    {
        Tensor<Scalar>(
            randomTruncatedNormal: $0,
            mean: mean,
            standardDeviation: standardDeviation,
            seed: seed)
    }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
    /// Creates a tensor with the specified shape, randomly sampling scalar values from a truncated
    /// Normal distribution.
    ///
    /// - Parameters:
    ///   - shape: The dimensions of the tensor.
    ///   - mean: The mean of the distribution.
    ///   - standardDeviation: The standard deviation of the distribution.
    ///   - seed: The seed value.
    public init(
        randomTruncatedNormal shape: TensorShape,
        mean: Tensor<Scalar> = Tensor<Scalar>(0),
        standardDeviation: Tensor<Scalar> = Tensor<Scalar>(1),
        seed: TensorFlowSeed = TensorFlow.Context.local.randomSeed
    ) {
        let sample: Tensor<Scalar> = _Raw.statelessTruncatedNormal(
            shape: Tensor<Int32>((0..<shape.rank).map { Int32(shape[$0]) }),
            seed: Tensor<Int32>([seed.graph, seed.op]))
        self = standardDeviation * sample + mean
    }
}
