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

#if os(iOS) || os(macOS) || os(tvOS) || os(watchOS)
  import Darwin
#elseif os(Windows)
  #if canImport(CRT)
    import CRT
  #else
    import MSVCRT
  #endif
#else
  import Glibc
#endif

/// Returns a single tensor containing the log of the sum of the exponentials
/// in `x`.
///
/// Used for numerical stability when dealing with very small values.
@differentiable
public func logSumExp(_ x: [Float]) -> Float {
  if x.count == 0 { return -Float.infinity}
  let maxVal = x.max()!
  let exps = x.map { exp($0 - maxVal) }
  return maxVal + log(exps.reduce(into: 0, +=))
}

@derivative(of: logSumExp)
public func vjpLogSumExp(_ x: [Float]) -> (
  value: Float,
  pullback: (Float) -> (Array<Float>.TangentVector)
) {
  func pb(v: Float) -> (Array<Float>.TangentVector) {
    if x.count == 0 { return Array<Float>.TangentVector([]) }
    let maxVal = x.max()!
    let exps = x.map { exp($0 - maxVal) }
    let sumExp = exps.reduce(into: 0, +=)
    return Array<Float>.TangentVector(exps.map{ v * $0 / sumExp })
  }
  return (logSumExp(x), pb)
}

/// Returns a single tensor containing the log of the sum of the exponentials
/// in `lhs` and `rhs`.
///
/// Used for numerical stability when dealing with very small values.
@differentiable
public func logSumExp(_ lhs: Float, _ rhs: Float) -> Float {
  return logSumExp([lhs, rhs])
}

/// A storage mechanism for scoring inside a lattice.
public struct SemiRing: Differentiable {

  /// The log likelihood.
  public var logp: Float

  /// The regularization factor.
  public var logr: Float

  /// Creates an instance with log likelihood `logp` and regularization
  /// factor `logr`.
  @differentiable
  public init(logp: Float, logr: Float) {
    self.logp = logp
    self.logr = logr
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
    let diffP = abs(self.logp - other.logp)
    let diffR = abs(self.logp - other.logp)

    return (diffP <= tolerance || diffP.isNaN)
      && (diffR <= tolerance || diffR.isNaN)
  }
}
