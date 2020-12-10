import TensorFlow

/// Metrics that can be registered into TrainingLoop.
public enum TrainingMetrics {
  case loss
  case accuracy
  case matthewsCorrelationCoefficient
  case perplexity

  public var name: String {
    switch self {
    case .loss:
      return "loss"
    case .accuracy:
      return "accuracy"
    case .matthewsCorrelationCoefficient:
      return "mcc"
    case .perplexity:
      return "ppl"
    }
  }

  public var measurer: MetricsMeasurer {
    switch self {
    case .loss:
      return LossMeasurer(self.name)
    case .accuracy:
      return AccuracyMeasurer(self.name)
    case .matthewsCorrelationCoefficient:
      return MCCMeasurer(self.name)
    case .perplexity:
      return PerplexityMeasurer(self.name)
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

/// A measurer for measuring matthewsCorrelationCoefficient.
public struct MCCMeasurer: MetricsMeasurer {
  /// Name of the MCCMeasurer.
  public var name: String

  /// A collection of predicted values.
  private var predictions: [Bool] = []

  /// A collection of ground truth values.
  private var groundTruths: [Bool] = []

  /// Creates an instance of MCCMeasurer named `name`.
  public init(_ name: String = "mcc") {
    self.name = name
  }

  /// Empties self.predictions and self.groundTruths.
  public mutating func reset() {
    predictions = []
    groundTruths = []
  }

  /// Appends boolean values computed from `predictions` and `labels` 
  /// to self.predictions and self.groundTruths.
  public mutating func accumulate<Output, Target>(
    loss: Tensor<Float>?, predictions: Output?, labels: Target?
  ) {
    guard let logits = predictions as? Tensor<Float>, let labels = labels as? Tensor<Int32>
    else { return }

    self.predictions.append(contentsOf: (sigmoid(logits.flattened()) .>= 0.5).scalars)
    self.groundTruths.append(contentsOf: labels.scalars.map { $0 == 1 })
  }

  /// Computes the Matthews correlation coefficient.
  ///
  /// Note: The Matthews correlation coefficient is more informative than other confusion matrix measures
  /// (such as F1 score and accuracy) in evaluating binary classification problems, because it takes
  /// into account the balance ratios of the four confusion matrix categories (true positives, true
  /// negatives, false positives, false negatives).
  ///
  /// - Source: [https://en.wikipedia.org/wiki/Matthews_correlation_coefficient](
  ///             https://en.wikipedia.org/wiki/Matthews_correlation_coefficient).
  public func measure() -> Float {
    var tp = 0  // True positives.
    var tn = 0  // True negatives.
    var fp = 0  // False positives.
    var fn = 0  // False negatives.
    for (prediction, truth) in zip(predictions, groundTruths) {
      switch (prediction, truth) {
      case (false, false): tn += 1
      case (false, true): fn += 1
      case (true, false): fp += 1
      case (true, true): tp += 1
      }
    }
    let nominator = Float(tp * tn - fp * fn)
    let denominator = Float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).squareRoot()
    return denominator != 0 ? nominator / denominator : 0
  }
}

/// A measurer for measuring perplexity.
public struct PerplexityMeasurer: MetricsMeasurer {
  /// Name of the PerplexityMeasurer.
  public var name: String

  /// Sum of losses cumulated from batches.
  private var totalBatchLoss: Float = 0

  /// Count of batchs cumulated so far.
  private var batchCount: Int32 = 0

  /// Creates an instance with the PerplexityMeasurer named `name`.
  public init(_ name: String = "ppl") {
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

  /// Computes averaged perplexity as e^(averaged loss).
  public func measure() -> Float {
    return exp(totalBatchLoss / Float(batchCount))
  }
}
