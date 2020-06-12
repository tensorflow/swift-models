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

/// A structure used for scoring all possible segmentations of a character
/// sequence.
///
/// The path with the best score provides the most likely segmentation.
public struct Lattice: Differentiable {

  /// Represents a word.
  ///
  /// At each character position, an edge is constructed for every possible
  /// segmentation of the preceding portion of the sequence.
  public struct Edge: Differentiable {

    /// The node position immediately preceding this edge.
    @noDerivative public var start: Int

    /// The node position immediately following this edge.
    @noDerivative public var end: Int

    /// The characters composing a word.
    @noDerivative public var string: CharacterSequence

    /// The log likelihood of this segmentation.
    public var logp: Float

    /// The expected score for this segmentation.
    public var score: SemiRing

    /// The expected total score for this segmentation.
    public var totalScore: SemiRing

    /// Creates an edge for `sentence` between `start` and `end`.
    ///
    /// Uses the log probability `logp` and the power of the length penalty
    /// `order` to calculate the regularization factor and form the current
    /// score. Sums this score with `previous` to determine the total score.
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

    /// Creates an edge for `string` between `start` and `end` and sets the
    /// log probability `logp`, `score`, and `totalScore`.
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

  /// Represents a word boundary.
  ///
  /// When a lattice is built, a start node is created, followed by one for
  /// every character in the sequence, representing every potential boundary.
  ///
  /// - Note: Scores are only meaningful in relation to incoming edges and the
  ///   start node has no incoming edges.
  public struct Node: Differentiable {

    /// The incoming edge with the highest score.
    @noDerivative public var bestEdge: Edge?

    /// The score of the best incoming edge.
    public var bestScore: Float = 0.0

    /// All incoming edges.
    public var edges = [Edge]()

    /// A composite score of all incoming edges.
    public var semiringScore: SemiRing = SemiRing.one

    /// Creates an empty instance.
    init() {}

    /// Creates a node preceded by `bestEdge`, sets incoming edges to
    /// `edges`, and stores `bestScore` and `semiringScore`.
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

    /// Returns a sum of the total score of all incoming edges.
    @differentiable
    func computeSemiringScore() -> SemiRing {
      // TODO: Reduceinto and +=
      edges.differentiableMap { $0.totalScore }.sum()
    }

    /// Calculates and sets the current semiring score.
    @differentiable
    mutating func recomputeSemiringScore() {
      semiringScore = computeSemiringScore()
    }
  }

  /// Represents the position of word boundaries.
  var positions: [Node]

  /// Accesses the node at the `index`th position.
  @differentiable
  public subscript(index: Int) -> Node {
    get { return positions[index] }

    // TODO(TF-1193): Support derivative registration for accessors.
    // This enables cleanup:
    // - Before: `lattice.positions.update(at: i, to: node)`
    // - After: `lattice[i] = node`
    set { positions[index] = newValue }

    // _modify { yield &positions[index] }
  }

  /// Creates an empty instance with a start node, followed by `count` nodes.
  init(count: Int) {
    positions = Array(repeating: Node(), count: count + 1)
  }

  /// Creates an instance with the nodes in `positions`.
  public init(positions: [Node]) {
    self.positions = positions
  }

  /// Returns the path representing the best segmentation of `sentence`.
  public mutating func viterbi(sentence: CharacterSequence) -> [Edge] {
    // Forward pass
    // Starts at 1 since the 0 node has no incoming edges.
    for position in 1...sentence.count {
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

    // Backward pass
    var bestPath: [Edge] = []
    var nextEdge = self[sentence.count].bestEdge!
    while nextEdge.start != 0 {
      bestPath.append(nextEdge)
      nextEdge = self[nextEdge.start].bestEdge!
    }
    bestPath.append(nextEdge)

    return bestPath.reversed()
  }

  /// Returns the plain text encoded in `path`, using `alphabet`.
  ///
  /// This represents the segmentation of the full character sequence.
  public static func pathToPlainText(path: [Edge], alphabet: Alphabet) -> String {
    var plainText = [String]()
    for edge in path {
      for id in edge.string.characters {
        guard let character = alphabet.dictionary.key(id) else { continue }
        plainText.append(character)
      }
      plainText.append(" ")
    }
    return plainText.joined()
  }
}

extension Lattice: CustomStringConvertible {

  /// The plain text description of this instance that describes all nodes.
  public var description: String {
    """
    [
    \(positions.enumerated().map { "  \($0.0):  \($0.1)" }.joined(separator: "\n\n"))
    ]
    """
  }
}

extension Lattice.Node: CustomStringConvertible {

  /// The plain text description of this instance that describes all incoming
  /// edges.
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

  /// The plain text description of this instance with all edge details.
  public var description: String {
    "[\(start)->\(end)] logp: \(logp), score: \(score.shortDescription), total score: \(totalScore.shortDescription), sentence: \(string)"
  }
}

extension Lattice {

  /// Returns true when all nodes in `self` are within `tolerance` of all
  /// nodes in `other`.
  ///
  /// - Note: This behavior is modeled after SE-0259.
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

  /// Returns true when all properties and edges in `self` are within
  /// `tolerance` of all properties and edges in `other`.
  ///
  /// - Note: This behavior is modeled after SE-0259.
  public func isAlmostEqual(to other: Self, tolerance: Float) -> Bool {
    guard self.edges.count == other.edges.count else { return false }

    let diffBestScore = abs(self.bestScore - other.bestScore)
    if !(diffBestScore <= tolerance || diffBestScore.isNaN) {
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

  /// Returns true when the log likelihood and scores in `self` are within
  /// `tolerance` of the log likelihood and scores in `other`.
  ///
  /// - Note: This behavior is modeled after SE-0259.
  public func isAlmostEqual(to other: Self, tolerance: Float) -> Bool {
    let diffP = abs(self.logp - other.logp)
    return self.start == other.start && self.end == other.end
      // TODO: figure out why the string equality is being ignored
      // self.string == other.string &&
      && (diffP <= tolerance || diffP.isNaN)
      && self.score.isAlmostEqual(to: other.score, tolerance: tolerance)
      && self.totalScore.isAlmostEqual(to: other.totalScore, tolerance: tolerance)
  }
}
