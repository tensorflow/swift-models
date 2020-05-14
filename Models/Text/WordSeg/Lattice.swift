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

import ModelSupport
import TensorFlow

#if os(iOS) || os(macOS) || os(tvOS) || os(watchOS)
  import Darwin
#elseif os(Windows)
  import ucrt
#else
  import Glibc
#endif

/// Lattice
///
/// Represents the lattice used by the WordSeg algorithm.
public struct Lattice: Differentiable {
  /// Edge
  ///
  /// Represents an Edge
  public struct Edge: Differentiable {
    @noDerivative public var start: Int
    @noDerivative public var end: Int
    @noDerivative public var string: CharacterSequence
    public var logp: Float

    // expectation
    public var score: SemiRing
    public var totalScore: SemiRing

    @differentiable
    init(
      start: Int, end: Int, sentence: CharacterSequence, logp: Float,
      previous: SemiRing, order: Int
    ) {
      self.start = start
      self.end = end
      self.string = sentence
      self.logp = logp

      self.score =
        SemiRing(
          logp: logp,
          // TODO(abdulras): this should really use integeral pow
          logr: logp + logf(powf(Float(sentence.count), Float(order))))
      self.totalScore = self.score * previous
    }

    @differentiable
    public init(
      start: Int, end: Int, string: CharacterSequence, logp: Float,
      score: SemiRing, totalScore: SemiRing
    ) {
      self.start = start
      self.end = end
      self.string = string
      self.logp = logp
      self.score = score
      self.totalScore = totalScore
    }
  }

  /// Node
  ///
  /// Represents a node in the lattice
  public struct Node: Differentiable {
    @noDerivative public var bestEdge: Edge?
    public var bestScore: Float = 0.0
    public var edges = [Edge]()
    public var semiringScore: SemiRing = SemiRing.one

    init() {}

    @differentiable
    public init(
      bestEdge: Edge?, bestScore: Float, edges: [Edge],
      semiringScore: SemiRing
    ) {
      self.bestEdge = bestEdge
      self.bestScore = bestScore
      self.edges = edges
      self.semiringScore = semiringScore
    }

    @differentiable
    func computeSemiringScore() -> SemiRing {
      // TODO: Reduceinto and +=
      semiRingSum(edges.differentiableMap{ $0.totalScore })
    }
  }

  var positions: [Node]

  @differentiable
  public subscript(index: Int) -> Node {
    get { return positions[index] }
    set(v) { positions[index] = v }
    //_modify { yield &positions[index] }
  }

  init(count: Int) {
    positions = Array(repeating: Node(), count: count + 1)
  }

  public init(positions: [Node]) {
    self.positions = positions
  }

  mutating func viterbi(sentence: String) -> [Edge] {
    // Forwards pass
    for position in 0...sentence.count {
      var bestScore = -Float.infinity
      var bestEdge: Edge!
      for edge in self[position].edges {
        let score: Float = self[edge.start].bestScore + edge.logp
        if score > bestScore {
          bestScore = score
          bestEdge = edge
        }
      }
      self[position].bestScore = bestScore
      self[position].bestEdge = bestEdge
    }

    // Backwards
    var bestPath: [Edge] = []
    var nextEdge = self[sentence.count].bestEdge!
    while nextEdge.start != 0 {
      bestPath.append(nextEdge)
      nextEdge = self[nextEdge.start].bestEdge!
    }
    bestPath.append(nextEdge)

    return bestPath.reversed()
  }
}

extension Lattice: CustomStringConvertible {
  public var description: String {
    """
    [
    \(positions.enumerated().map { "  \($0.0):  \($0.1)" }.joined(separator: "\n\n"))
    ]
    """
  }
}

extension Lattice.Node: CustomStringConvertible {
  public var description: String {
    var edgesStr: String
    if edges.isEmpty {
      edgesStr = "    <no edges>"
    } else {
      edgesStr = edges.enumerated().map { "    \($0.0) - \($0.1)" }.joined(separator: "\n")
    }
    return """
      best edge: \(String(describing: bestEdge)), best score: \(bestScore), score: \(semiringScore.shortDescription)
      \(edgesStr)
      """
  }
}

extension Lattice.Edge: CustomStringConvertible {
  public var description: String {
    "[\(start)->\(end)] logp: \(logp), score: \(score.shortDescription), total score: \(totalScore.shortDescription), sentence: \(string)"
  }
}

/// SE-0259-esque equality with tolerance
extension Lattice {
  public func isAlmostEqual(to other: Self, tolerance: Float) -> Bool {
    guard self.positions.count == other.positions.count else {
      print("positions count mismatch: \(self.positions.count) != \(other.positions.count)")
      return false
    }
    return zip(self.positions, other.positions).enumerated()
      .map { (index, position) in
        let eq = position.0.isAlmostEqual(to: position.1, tolerance: tolerance)
        if !eq {
          print("mismatch at \(index): \(position.0) != \(position.1)")
        }
        return eq
      }
      .reduce(true) { $0 && $1 }
  }
}

extension Lattice.Node {
  public func isAlmostEqual(to other: Self, tolerance: Float) -> Bool {
    guard self.edges.count == other.edges.count else { return false }

    if !self.bestScore.isAlmostEqual(to: other.bestScore, tolerance: tolerance) {
      return false
    }
    if let lhs = self.bestEdge, let rhs = other.bestEdge {
      if !lhs.isAlmostEqual(to: rhs, tolerance: tolerance) {
        return false
      }
    }
    if !self.semiringScore.isAlmostEqual(to: other.semiringScore, tolerance: tolerance) {
      return false
    }
    return zip(self.edges, other.edges)
      .map { $0.isAlmostEqual(to: $1, tolerance: tolerance) }
      .reduce(true) { $0 && $1 }
  }
}

extension Lattice.Edge {
  public func isAlmostEqual(to other: Self, tolerance: Float) -> Bool {
    return self.start == other.start && self.end == other.end
      // TODO: figure out why the string equality is being ignored
      // self.string == other.string &&
      && self.logp.isAlmostEqual(to: other.logp, tolerance: tolerance)
      && self.score.isAlmostEqual(to: other.score, tolerance: tolerance)
      && self.totalScore.isAlmostEqual(to: other.totalScore, tolerance: tolerance)
  }
}
