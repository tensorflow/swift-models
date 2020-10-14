import TensorFlow

/// Metrics that can be registered into TrainingLoop.
public enum TrainingMetrics {
  case loss
  case accuracy

  public var name: String {
    switch self {
    case .loss:
      return "loss"
    case .accuracy:
      return "accuracy"
    }
  }

  public var measurer: MetricsMeasurer {
    switch self {
    case .loss:
      return LossMeasurer(self.name)
    case .accuracy:
      return AccuracyMeasurer(self.name)
    }
  }
}

/// An accumulator of statistics.
public protocol MetricsMeasurer {
  /// Name of the metrics.
  var name: String { get set }

  /// Clears accumulated data up and resets measurer to initial state.
  mutating func reset()

  /// Accumulates data from `loss`, `predictions`, `labels`.
  mutating func accumulate<Output, Target>(
    loss: Tensor<Float>?, predictions: Output?, labels: Target?
  )

  /// Computes metrics from cumulated data.
  func measure() -> Float
}

/// A measurer for measuring loss.
public struct LossMeasurer: MetricsMeasurer {
  /// Name of the LossMeasurer.
  public var name: String

  /// Sum of losses cumulated from batches.
  private var totalBatchLoss: Float = 0

  /// Count of batchs cumulated so far.
  private var batchCount: Int32 = 0

  /// Creates an instance with the LossMeasurer named `name`.
  public init(_ name: String = "loss") {
    self.name = name
  }

  /// Resets totalBatchLoss and batchCount to zero.
  public mutating func reset() {
    totalBatchLoss = 0
    batchCount = 0
  }

  /// Adds `loss` to totalBatchLoss and increases batchCount by one.
  public mutating func accumulate<Output, Target>(
    loss: Tensor<Float>?, predictions: Output?, labels: Target?
  ) {
    if let newBatchLoss = loss {
      totalBatchLoss += newBatchLoss.scalarized()
      batchCount += 1
    }
  }

  /// Computes averaged loss.
  public func measure() -> Float {
    return totalBatchLoss / Float(batchCount)
  }
}

/// A measurer for measuring accuracy
public struct AccuracyMeasurer: MetricsMeasurer {
  /// Name of the AccuracyMeasurer.
  public var name: String

  /// Count of correct guesses.
  private var correctGuessCount: Int32 = 0

  /// Count of total guesses.
  private var totalGuessCount: Int32 = 0

  /// Creates an instance with the AccuracyMeasurer named `name`. 
  public init(_ name: String = "accuracy") {
    self.name = name
  }

  /// Resets correctGuessCount and totalGuessCount to zero.
  public mutating func reset() {
    correctGuessCount = 0
    totalGuessCount = 0
  }

  /// Computes correct guess count from `loss`, `predictions` and `labels`
  /// and adds it to correctGuessCount; Computes total guess count from
  /// `labels` shape and adds it to totalGuessCount.
  public mutating func accumulate<Output, Target>(
    loss: Tensor<Float>?, predictions: Output?, labels: Target?
  ) {
    guard let predictions = predictions as? Tensor<Float>, let labels = labels as? Tensor<Int32>
    else {
      fatalError(
        "For accuracy measurements, the model output must be Tensor<Float>, and the labels must be Tensor<Int>."
      )
    }
    correctGuessCount += Tensor<Int32>(predictions.argmax(squeezingAxis: -1) .== labels).sum()
      .scalarized()
    totalGuessCount += Int32(labels.shape.reduce(1, *))
  }

  /// Computes accuracy as percentage of correct guesses.
  public func measure() -> Float {
    return Float(correctGuessCount) / Float(totalGuessCount)
  }
}
