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

public protocol Batchable {
  func flattenedBatch(outerDimCount: Int) -> Self
  func unflattenedBatch(outerDims: [Int]) -> Self
}

public protocol DifferentiableBatchable: Batchable, Differentiable {
  @differentiable(wrt: self)
  func flattenedBatch(outerDimCount: Int) -> Self

  @differentiable(wrt: self)
  func unflattenedBatch(outerDims: [Int]) -> Self
}

extension Tensor: Batchable {
  public func flattenedBatch(outerDimCount: Int) -> Tensor {
    if outerDimCount == 1 {
      return self
    }
    var newShape = [-1]
    for i in outerDimCount..<rank {
      newShape.append(shape[i])
    }
    return reshaped(to: TensorShape(newShape))
  }

  public func unflattenedBatch(outerDims: [Int]) -> Tensor {
    if rank > 1 {
      return reshaped(to: TensorShape(outerDims + shape.dimensions[1...]))
    }
    return reshaped(to: TensorShape(outerDims))
  }
}

extension Tensor: DifferentiableBatchable where Scalar: TensorFlowFloatingPoint {
  @differentiable(wrt: self)
  public func flattenedBatch(outerDimCount: Int) -> Tensor {
    if outerDimCount == 1 {
      return self
    }
    var newShape = [-1]
    for i in outerDimCount..<rank {
      newShape.append(shape[i])
    }
    return reshaped(to: TensorShape(newShape))
  }

  @differentiable(wrt: self)
  public func unflattenedBatch(outerDims: [Int]) -> Tensor {
    if rank > 1 {
      return reshaped(to: TensorShape(outerDims + shape.dimensions[1...]))
    }
    return reshaped(to: TensorShape(outerDims))
  }
}

public protocol Distribution {
  associatedtype Value

  func entropy() -> Tensor<Float>

  /// Returns a random sample drawn from this distribution.
  func sample() -> Value
}

public protocol DifferentiableDistribution: Distribution, Differentiable {
  @differentiable(wrt: self)
  func entropy() -> Tensor<Float>
}

public struct Categorical<Scalar: TensorFlowIndex>: DifferentiableDistribution, KeyPathIterable {
  /// Log-probabilities of this categorical distribution.
  public var logProbabilities: Tensor<Float>

  @inlinable  
  @differentiable(wrt: probabilities)
  public init(probabilities: Tensor<Float>) {
    self.logProbabilities = log(probabilities)
  }

  @inlinable
  @differentiable(wrt: self)
  public func entropy() -> Tensor<Float> {
    -(logProbabilities * exp(logProbabilities)).sum(squeezingAxes: -1)
  }

  @inlinable
  public func sample() -> Tensor<Scalar> {
    let seed = Context.local.randomSeed
    let outerDimCount = self.logProbabilities.rank - 1
    let logProbabilities = self.logProbabilities.flattenedBatch(outerDimCount: outerDimCount)
    let multinomial: Tensor<Scalar> = _Raw.multinomial(
      logits: logProbabilities,
      numSamples: Tensor<Int32>(1),
      seed: Int64(seed.graph),
      seed2: Int64(seed.op))
    let flattenedSamples = multinomial.gathering(atIndices: Tensor<Int32>(0), alongAxis: 1)
    return flattenedSamples.unflattenedBatch(
      outerDims: [Int](self.logProbabilities.shape.dimensions[0..<outerDimCount]))
  }
}
