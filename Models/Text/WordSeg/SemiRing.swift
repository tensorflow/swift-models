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

#if os(iOS) || os(macOS) || os(tvOS) || os(watchOS)
  import Darwin
#elseif os(Windows)
  import ucrt
#else
  import Glibc
#endif

/// Returns a single tensor containing the log of the sum of the exponentials
/// in `x`.
///
/// Used for numerical stability when dealing with very small values.
@differentiable
public func logSumExp(_ x: [Tensor<Float>]) -> Tensor<Float> {
  // Deal with an empty array first.
  if x.count == 0 { return Tensor(-Float.infinity) }
  return Tensor<Float>(stacking: x).logSumExp()
}

/// Returns a single tensor containing the log of the sum of the exponentials
/// in `lhs` and `rhs`.
///
/// Used for numerical stability when dealing with very small values.
@differentiable
public func logSumExp(_ lhs: Tensor<Float>, _ rhs: Tensor<Float>) -> Tensor<Float> {
  return logSumExp([lhs, rhs])
}

/// A storage mechanism for scoring inside a lattice.
public struct SemiRing: Differentiable {

  /// The log likelihood.
  public var logp: Tensor<Float>

  /// The regularization factor.
  public var logr: Tensor<Float>

  /// Creates an instance with log likelihood `logp` and regularization
  /// factor `logr`.
  @differentiable
  public init(logp: Tensor<Float>, logr: Tensor<Float>) {
    self.logp = logp
    self.logr = logr
  }

  /// Creates an instance with log likelihood `logp` and regularization
  /// factor `logr`.
  @differentiable
  public init(logp: Float, logr: Float) {
    self.logp = Tensor(logp)
    self.logr = Tensor(logr)
  }

  /// The baseline score of zero.
  static var zero: SemiRing { SemiRing(logp: -Float.infinity, logr: -Float.infinity) }

  /// The baseline score of one.
  static var one: SemiRing { SemiRing(logp: 0.0, logr: -Float.infinity) }
}

/// Multiplies `lhs` by `rhs`.
///
/// Since scores are on a logarithmic scale, products become sums.
@differentiable
func * (_ lhs: SemiRing, _ rhs: SemiRing) -> SemiRing {
  return SemiRing(
    logp: lhs.logp + rhs.logp,
    logr: logSumExp(lhs.logp + rhs.logr, rhs.logp + lhs.logr))
}

/// Sums `lhs` by `rhs`.
@differentiable
func + (_ lhs: SemiRing, _ rhs: SemiRing) -> SemiRing {
  return SemiRing(
    logp: logSumExp(lhs.logp, rhs.logp),
    logr: logSumExp(lhs.logr, rhs.logr))
}

extension Array where Element == SemiRing {

  /// Returns a sum of all scores in the collection.
  @differentiable
  func sum() -> SemiRing {
    return SemiRing(
      logp: logSumExp(differentiableMap { $0.logp }),
      logr: logSumExp(differentiableMap { $0.logr }))
  }
}

extension SemiRing {

  /// The plain text description of this instance with score details.
  var shortDescription: String {
    "(\(logp), \(logr))"
  }
}

extension SemiRing {

  /// Returns true when `self` is within `tolerance` of `other`.
  ///
  /// - Note: This behavior is modeled after SE-0259.
  // TODO(abdulras) see if we can use ulp as a default tolerance
  @inlinable
  public func isAlmostEqual(to other: Self, tolerance: Float) -> Bool {
    return
      (self.logp.isAlmostEqual(to: other.logp, tolerance: tolerance)
      || (self.logp.scalarized().isInfinite && other.logp.scalarized().isInfinite))
      && (self.logr.isAlmostEqual(to: other.logr, tolerance: tolerance)
        || (self.logr.scalarized().isInfinite && other.logr.scalarized().isInfinite))
  }
}
