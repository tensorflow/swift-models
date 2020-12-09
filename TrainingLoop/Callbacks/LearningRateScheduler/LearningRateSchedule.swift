/// The abstract shape of a function mapping `Float` into `Float`
public struct Shape {
  /// Creates an instance representing the shape of the values of `curve` over the domain 0...1.
  public init(_ curve: @escaping (Float) -> Float) {
    self.curve = curve
    curveSampledAt.0 = curve(0)
    curveSampledAt.1 = curve(1)
    precondition(
      curveSampledAt.0 != curveSampledAt.1,
      "Curve must have distinct values at 0 and 1.")
  }

  /// Returns a function `f` such that `f(domain.lowerBound) == startResult` and
  /// `f(domain.upperBound) == endResult`, with the shape of intermediate values defined by `self`.
  ///
  /// The `curve` with which `self` was created is sampled between 0 and 1 and scaled linearly to
  /// fit the constraints.
  public func projected(
    intoDomain domain: ClosedRange<Float>, startResult: Float, endResult: Float
  )
    -> (Float) -> Float
  {
    if domain.lowerBound == domain.upperBound {
      // This kind of projection can be useful for modeling discontinuities.
      return {
          x in
          x == domain.lowerBound ? startResult : endResult
        }
    }
    let domainSize = domain.upperBound - domain.lowerBound
    let rangeScale = (endResult - startResult) / (curveSampledAt.1 - curveSampledAt.0)
    return { x in
      let domainFraction = (x - domain.lowerBound) / domainSize
      return (curve(domainFraction) - curveSampledAt.0) * rangeScale + startResult
    }
  }

  /// A function describing the shape, when sampled between 0.0 and 1.0
  private let curve: (Float) -> Float

  /// Samples of `curve` taken at 0 and 1.
  private let curveSampledAt: (Float, Float)
}

/// A mapping from step number to learning rate, expressed as a collection of learning rates, and as
/// a callable “function instance”.
public struct LearningRateSchedule {
  /// A fragment of the schedule, when paired with known start step
  fileprivate typealias Segment = (endStep: Int, rateAtStep: (Int) -> Float)

  /// The entire representation of self.
  ///
  /// Always contains at least one segment with an endStep of 0.
  private var segments: [Segment]

  /// Creates a schedule that begins at `startRate`.
  public init(startRate: Float) {
    segments = [
      (
        endStep: 0,
        rateAtStep: { _ in
          startRate
        }
      ),
    ]
  }

  /// Returns the learning rate at step `n`.
  ///
  /// For the step that connects the neighbor segments, always return the 
  /// value at the previous segment. 
  ///
  /// - Precondition: `n >= 0 && n < count`
  /// - Complexity: O(log(count))
  public func callAsFunction(_ n: Int) -> Float {
    precondition(n >= 0)
    precondition(n < count)
    let p = segments.partitionPoint { $0.endStep >= n }
    return segments[p].rateAtStep(n)
  }

  /// Appends an `n`-step segment with shape `s` start rate `startRate` and end rate `endRate`
  ///
  /// The start rate of the appended segment is `startRate` if provided, or the end rate of the last 
  /// segment appended, or if `self.isEmpty`, the `startRate` with which `self` was initialized.
  ///
  /// If there is a discontinous rate jump at connecting step of neighbor segments, use rate of
  /// the first segment as the rate at that step, and that of the second segment is only used to  
  /// compute rates at later steps in second segment.
  ///
  /// - Precondition: n > 0
  public mutating func appendSegment(
    stepCount n: Int, shape s: Shape, startRate: Float? = nil, endRate: Float
  ) {
    precondition(n >= 0)
    let newEnd = count - 1 + n - 1
    let lastRate: Float = segments.last!.rateAtStep(segments.last!.endStep)
    let curve = s.projected(
      intoDomain: Float(count - 1)...Float(newEnd),
      startResult: startRate ?? lastRate, endResult: endRate)
    segments.append((endStep: newEnd, rateAtStep: { curve(Float($0)) }))
  }
}

extension LearningRateSchedule: BidirectionalCollection {
  /// An element position.
  public struct Index {
    /// The absolute step number of the element at `self`.
    fileprivate let step: Int

    /// The position of the segment that generates the element at `self`.
    fileprivate let segment: Array<Segment>.Index
  }

  /// The position of the first element.
  public var startIndex: Index { Index(step: 0, segment: 0) }

  /// The position one past the last element.
  public var endIndex: Index { Index(step: count, segment: segments.count) }

  /// The number of elements in `self`.
  public var count: Int { segments.last!.endStep + 1 }

  /// Returns the element at `i`.
  public subscript(i: Index) -> Float {
    return segments[i.segment].rateAtStep(i.step)
  }

  /// Returns the position in `self` following `i`.
  public func index(after i: Index) -> Index {
    let newStep = i.step + 1
    let newSegment = i.segment + (newStep > segments[i.segment].endStep ? 1 : 0)
    return Index(step: newStep, segment: newSegment)
  }

  /// Returns the position in `self` preceding `i`.
  public func index(before i: Index) -> Index {
    let newSegment = i.segment - (i.step == segments[i.segment - 1].endStep ? 1 : 0)
    return Index(step: i.step - 1, segment: newSegment)
  }
}

extension LearningRateSchedule.Index: Comparable {
  public static func == (l: Self, r: Self) -> Bool { return l.step == r.step }
  public static func < (l: Self, r: Self) -> Bool { return l.step < r.step }
}

extension Collection {
  /// Returns the index of the first element that matches the predicate.
  ///
  /// The collection must already be partitioned according to the predicate, as if
  /// `self.partition(by: predicate)` had already been called.
  func partitionPoint(
    where predicate: (Element) throws -> Bool
  ) rethrows -> Index {
    var n = distance(from: startIndex, to: endIndex)
    var l = startIndex

    while n > 0 {
      let half = n / 2
      let mid = index(l, offsetBy: half)
      if try predicate(self[mid]) {
        n = half
      } else {
        l = index(after: mid)
        n -= half + 1
      }
    }
    return l
  }
}
